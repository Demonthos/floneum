use crate::raw::cache::LlamaCache;
use crate::token_stream::TokenOutputStream;
use crate::token_stream::TokenOutputStreamError;
use crate::{raw::Model, session::LlamaSession};
use kalosm_common::*;
use kalosm_language_model::GenerationParameters;
use llm_samplers::types::Logits;
use std::any::Any;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use candle_core::{
    quantized::{ggml_file, gguf_file},
    DType, Device,
};
use tokenizers::Tokenizer;

use crate::{InferenceSettings, LlamaSourceError};

/// An error that can occur when running a [`LlamaModel`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// An error from candle while running the model.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// An error from tokenizers while running the model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(tokenizers::Error),

    /// An error while sampling tokens.
    #[error("Sampler error: {0}")]
    SamplerError(Box<dyn std::error::Error + Send + Sync>),

    /// A streaming detokenization error.
    #[error("Token output stream error: {0}")]
    TokenOutputStreamError(TokenOutputStreamError),

    /// An error while writing to the session cache.
    #[error("Session cache error: {0}")]
    Session(String),

    /// No valid tokens were sampled during structured generation
    #[error("No valid tokens were sampled")]
    NoValidTokens,

    /// The model has already stopped.
    #[error("Model stopped")]
    ModelStopped,
}

/// The inner, synchronous Llama model.
pub struct LlamaModel {
    pub(crate) model: Model,
    pub(crate) device: Device,
    pub(crate) tokenizer: Arc<Tokenizer>,
}

impl LlamaModel {
    pub(crate) fn forward(
        model: &Model,
        device: &Device,
        tokens: &[u32],
        cache: Option<&mut LlamaCache>,
        logits_vec: &mut Vec<f32>,
    ) -> candle_core::Result<()> {
        if tokens.is_empty() {
            candle_core::bail!("Cannot run model on empty input");
        }

        let logits = model.forward(tokens, device, cache)?;

        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        copy_tensor_into_vec(&logits, logits_vec)?;

        Ok(())
    }

    /// Create a new sync Llama model from a builder.
    pub async fn from_builder(
        builder: crate::LlamaBuilder,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self, LlamaSourceError> {
        let device = builder.get_device()?;

        let tokenizer_source = format!("Tokenizer ({})", builder.source.tokenizer);
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer = builder
            .source
            .tokenizer(|progress| handler(create_progress(progress)))
            .await?;

        let source = format!("Model ({})", builder.source.model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let filename = builder
            .source
            .model(|progress| handler(create_progress(progress)))
            .await?;
        let mut file = std::fs::File::open(&filename)
            .expect("The path returned by LlamaSource::model should be valid");
        let model = match filename.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model = gguf_file::Content::read(&mut file)?;
                Model::from_gguf(model, &mut file, &device)?
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file, &device)?;
                let gqa = builder.source.group_query_attention;
                let vocab = tokenizer.get_vocab(true);
                let stop_token = match vocab
                    .get("</s>")
                    .or_else(|| vocab.get("<|end_of_text|>"))
                    .or_else(|| vocab.get("<|endoftext|>"))
                {
                    Some(token) => *token,
                    None => return Err(LlamaSourceError::NoStopToken),
                };
                Model::from_ggml(model, gqa as usize, &device, stop_token)?
            }
        };

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(model: Model, tokenizer: Arc<Tokenizer>, device: Device) -> Self {
        Self {
            model,
            device,
            tokenizer,
        }
    }

    pub(crate) fn _infer(
        &mut self,
        settings: InferenceSettings,
        mut on_token: Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>,
        finished: &tokio::sync::oneshot::Sender<Result<Box<dyn Any + Send>, LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let InferenceSettings {
            prompt,
            stop_on,
            mut sampler,
            session,
        } = settings;

        let mut session = session
            .cache
            .write()
            .map_err(|err| LlamaModelError::Session(err.to_string()))?;

        let tokens = self
            .tokenizer
            .encode(prompt, false)
            .map_err(LlamaModelError::Tokenizer)?;
        let tokens = tokens.get_ids();
        let mut text_stream = TokenOutputStream::new(self.tokenizer.clone());
        for &token in tokens {
            text_stream
                .next_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
        }

        let mut logit_probs = Vec::new();
        Self::forward(
            &self.model,
            &self.device,
            tokens,
            Some(&mut session),
            &mut logit_probs,
        )?;
        let mut logits = Logits::try_from_iter_top_k(logit_probs, 512)
            .expect("model output should be valid logits");
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.as_ref().map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_token = self.model.config.stop_token;
        let mut logit_probs = Vec::new();

        'generate: while !finished.is_closed() {
            let new_token = text_stream
                .sample_token(&mut sampler, logits, stop_on.as_deref())
                .map_err(LlamaModelError::TokenOutputStreamError)?;
            if new_token == stop_token {
                tracing::trace!("Stopping on stop token");
                break;
            }
            if let Some(mut new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                if let Some(stop_on) = stop_on_lowercase {
                    let lowercase = new_text.to_lowercase();

                    // Check if the string ends with the start of the stop_on string
                    let mut before_stop_on = None;
                    let remaining_stop_on = stop_on
                        .strip_prefix(&queued_text_matching_stop_on)
                        .unwrap_or(stop_on);

                    // If the remaining stop_on string is empty, we have found a match
                    if remaining_stop_on.is_empty() {
                        break;
                    }

                    for (i, _) in lowercase.char_indices() {
                        let end_of_new_text = &lowercase[i..];
                        if end_of_new_text.is_empty() {
                            break;
                        }

                        // Check if we have matched all of the stop_on string
                        if end_of_new_text.starts_with(remaining_stop_on) {
                            queued_text_matching_stop_on += end_of_new_text;
                            break 'generate;
                        }

                        // Check if the string ends with the start of the stop_on string
                        if remaining_stop_on.starts_with(end_of_new_text) {
                            before_stop_on = Some(lowercase[..i].to_string());
                            queued_text_matching_stop_on += end_of_new_text;
                            break;
                        }
                    }

                    match before_stop_on {
                        Some(before_stop_on) => {
                            on_token(before_stop_on)?;
                        }
                        None => {
                            new_text =
                                std::mem::take(&mut queued_text_matching_stop_on) + &new_text;
                            on_token(new_text)?;
                        }
                    }
                } else {
                    on_token(new_text)?;
                }
            }
            Self::forward(
                &self.model,
                &self.device,
                &[new_token],
                Some(&mut session),
                &mut logit_probs,
            )?;
            logits = Logits::try_from_iter_top_k(logit_probs.iter().copied(), 512)
                .expect("model output should be valid logits");
        }

        // Flush the queued text
        if let Some(stop_string) = stop_on_lowercase {
            if !queued_text_matching_stop_on.starts_with(stop_string) {
                on_token(queued_text_matching_stop_on)?;
            }
        }

        Ok(())
    }
}

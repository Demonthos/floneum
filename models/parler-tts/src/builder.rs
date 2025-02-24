use kalosm_model_types::{FileSource, ModelLoadingProgress};
use std::{collections::HashSet, sync::Arc};

use crate::{model::ParlerInner, Parler, ParlerDrop, ParlerLoadingError, ParlerMessage};

/// The Parler source model to use.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ParlerSource {
    /// An 880M parameter model.
    MiniV1,
    /// A 2.3B parameter model.
    #[default]
    LargeV1,
}

impl ParlerSource {
    pub(crate) fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::MiniV1 => ("parler-tts/parler-tts-mini-v1", "main"),
            Self::LargeV1 => ("parler-tts/parler-tts-large-v1", "main"),
        }
    }
}

pub(crate) struct ParlerModelConfig {
    model: FileSource,
    tokenizer: FileSource,
    config: FileSource,
}

impl ParlerModelConfig {
    pub(crate) fn new(model: FileSource, tokenizer: FileSource, config: FileSource) -> Self {
        Self {
            model,
            tokenizer,
            config,
        }
    }
}

/// A builder for the [`Parler`](crate::Parler) model.
#[derive(Debug, Default)]
pub struct ParlerBuilder {
    /// The model to be used.
    model: ParlerSource,

    /// The cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    cache: kalosm_common::Cache,
}

impl ParlerBuilder {
    /// Create a new [`ParlerBuilder`] with the default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model to be used.
    pub fn with_source(mut self, source: ParlerSource) -> Self {
        self.model = source;
        self
    }

    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;
        self
    }

    fn get_model_config(&self) -> ParlerModelConfig {
        let (model_id, revision) = self.model.model_and_revision();
        let model_file = match self.model {
            ParlerSource::MiniV1 => "model.safetensors",
            ParlerSource::LargeV1 => "model.safetensors.index.json",
        };

        let model = FileSource::huggingface(model_id, revision, model_file);
        let tokenizer = FileSource::huggingface(model_id, revision, "tokenizer.json");
        let config = FileSource::huggingface(model_id, revision, "config.json");

        ParlerModelConfig::new(model, tokenizer, config)
    }

    /// Build the model.
    pub async fn build(self) -> Result<Parler, ParlerLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a loading handler.
    pub async fn build_with_loading_handler(
        self,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Parler, ParlerLoadingError> {
        // Download files
        let sources = self.get_model_config();
        let tokenizer_source = sources.tokenizer;
        let model_source = sources.model;
        let config_source = sources.config;

        // Tokenizer progress
        let display_tokenizer_source = format!("Tokenizer ({})", tokenizer_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(display_tokenizer_source);
        let tokenizer_filename = self
            .cache
            .get(&tokenizer_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        // Model source progress
        let display_model_source = format!("Model ({})", model_source);
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_model_source);
        let model_file = self
            .cache
            .get(&model_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        // Determine if there are multiple model files and download them if required.
        let mut model_files = Vec::new();
        match self.model {
            ParlerSource::MiniV1 => model_files.push(model_file),
            // LargeV1 uses a weight map index file.
            ParlerSource::LargeV1 => {
                let json: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(model_file).unwrap())
                        .map_err(|e| ParlerLoadingError::ParseModel(e.into()))?;

                let weight_map = match json.get("weight_map") {
                    Some(serde_json::Value::Object(map)) => map,
                    _ => {
                        return Err(ParlerLoadingError::ParseModel(
                            "expected `weight_map` to be an object".into(),
                        ))
                    }
                };

                // De-deplicate files
                let mut file_names = HashSet::new();
                for value in weight_map.values() {
                    if let Some(file) = value.as_str() {
                        file_names.insert(file);
                    }
                }

                // Download the weight map files form the index.
                for file in file_names {
                    let (model_id, revision) = self.model.model_and_revision();
                    let source = FileSource::huggingface(model_id, revision, file);

                    let display_model_file = format!("Model ({})", source);
                    let mut create_progress =
                        ModelLoadingProgress::downloading_progress(display_model_file);

                    let model_file = self
                        .cache
                        .get(&source, |progress| {
                            progress_handler(create_progress(progress))
                        })
                        .await?;

                    model_files.push(model_file);
                }
            }
        };

        // Config progress
        let display_config_source = format!("Config ({})", config_source);
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_config_source);
        let config = self
            .cache
            .get(&config_source, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        // Start the parler thread
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let mut model = ParlerInner::new(model_files, tokenizer_filename, config).unwrap();
            while let Ok(message) = rx.recv() {
                match message {
                    ParlerMessage::Kill => return,
                    ParlerMessage::Generate {
                        settings,
                        prompt,
                        description,
                        result,
                    } => {
                        _ = result.send(model.generate(settings, prompt, description));
                    }
                }
            }
        });

        Ok(Parler {
            inner: Arc::new(ParlerDrop {
                thread: Some(thread),
                sender: tx,
            }),
        })
    }
}

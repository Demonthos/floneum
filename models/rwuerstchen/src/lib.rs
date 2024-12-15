//! # RWuerstchen
//!
//! RWuerstchen is a rust wrapper for library for [Wuerstchen](https://huggingface.co/papers/2306.00637) implemented in the [Candle](https://github.com/huggingface/candle) ML framework.
//!
//! RWuerstchen generates images efficiently from text prompts.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use futures_util::StreamExt;
//! use rwuerstchen::*;
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     let model = Wuerstchen::builder().build().await?;
//!     let settings = WuerstchenInferenceSettings::new(
//!         "a cute cat with a hat in a room covered with fur with incredible detail",
//!     );
//!     if let Ok(mut images) = model.run(settings) {
//!         while let Some(image) = images.next().await {
//!             if let Some(buf) = image.generated_image() {
//!                 buf.save(&format!("{}.png",image.sample_num()))?;
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

use std::{sync::OnceLock, time::Duration};

use futures_util::{Stream, StreamExt};
use image::ImageBuffer;
pub use kalosm_common::ModelLoadingProgress;
use kalosm_common::{CacheError, FileSource};
use kalosm_language_model::ModelBuilder;

use kalosm_streams::text_stream::ChannelImageStream;
use model::{WuerstcheModelSettings, WuerstchenInner};
use Result;

mod model;

static ZERO_IMAGE: OnceLock<ImageBuffer<image::Rgb<u8>, Vec<u8>>> = OnceLock::new();

#[derive(Debug, Clone)]
struct DiffusionResult {
    image: ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    height: usize,
    width: usize,
}

/// An image generated by the model
#[derive(Debug)]
pub struct Image {
    sample_num: i64,
    elapsed_time: Duration,
    remaining_time: Duration,
    progress: f32,
    result: candle_core::Result<DiffusionResult>,
}

impl Image {
    /// Get the sample number
    pub fn sample_num(&self) -> i64 {
        self.sample_num
    }

    /// Get the elapsed time
    pub fn elapsed_time(&self) -> Duration {
        self.elapsed_time
    }

    /// Get the estimated time remaining to process the entire audio file
    pub fn remaining_time(&self) -> Duration {
        self.remaining_time
    }

    /// The progress of the transcription, from 0 to 1
    pub fn progress(&self) -> f32 {
        self.progress
    }

    /// Get the height in px of the generated image
    pub fn height(&self) -> Option<usize> {
        self.result.as_ref().ok().map(|val| val.height)
    }

    /// Get the width in px of the generated image
    pub fn width(&self) -> Option<usize> {
        self.result.as_ref().ok().map(|val| val.width)
    }

    /// Get the generated image
    pub fn generated_image(&self) -> Option<ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        self.result.as_ref().ok().map(|val| val.image.clone())
    }

    /// Get the error message if no image has been generated
    pub fn error(&self) -> Option<&candle_core::Error> {
        self.result.as_ref().err()
    }
}

impl AsRef<ImageBuffer<image::Rgb<u8>, Vec<u8>>> for Image {
    fn as_ref(&self) -> &ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        match &self.result {
            Ok(val) => &val.image,
            Err(_) => ZERO_IMAGE.get_or_init(|| ImageBuffer::new(0, 0)),
        }
    }
}

/// An extension trait for transcribing audio streams.
pub trait GenerateImageStreamExt {
    /// Transcribe the audio stream.
    fn generate(self, model: Wuerstchen) -> ChannelImageStream<Image>;
}

impl<S> GenerateImageStreamExt for S
where
    S: Stream<Item = WuerstchenInferenceSettings> + std::marker::Unpin + Send + 'static,
{
    fn generate(self, model: Wuerstchen) -> ChannelImageStream<Image> {
        let mut stream = self;
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(source) = stream.next().await {
                let mut stream = { model.run(source) };
                while let Some(segment) = stream.next().await {
                    if let Err(err) = sender.send(segment) {
                        tracing::error!("error sending segment: {}", err);
                        return;
                    }
                }
            }
        });
        ChannelImageStream::from(receiver)
    }
}

/// A builder for the Wuerstchen model.
pub struct WuerstchenBuilder {
    use_flash_attn: bool,

    /// The decoder weight file, in .safetensors format.
    decoder_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    clip_weights: Option<String>,

    /// The CLIP weight file used by the prior model, in .safetensors format.
    prior_clip_weights: Option<String>,

    /// The prior weight file, in .safetensors format.
    prior_weights: Option<String>,

    /// The VQGAN weight file, in .safetensors format.
    vqgan_weights: Option<String>,

    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The file specifying the tokenizer to used for prior tokenization.
    prior_tokenizer: Option<String>,
}

impl Default for WuerstchenBuilder {
    fn default() -> Self {
        Self {
            use_flash_attn: { cfg!(feature = "flash") },
            decoder_weights: None,
            clip_weights: None,
            prior_clip_weights: None,
            prior_weights: None,
            vqgan_weights: None,
            tokenizer: None,
            prior_tokenizer: None,
        }
    }
}

impl WuerstchenBuilder {
    /// Set whether to use the Flash Attention implementation.
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    /// Set the decoder weight file, in .safetensors format.
    pub fn with_decoder_weights(mut self, decoder_weights: impl Into<String>) -> Self {
        self.decoder_weights = Some(decoder_weights.into());
        self
    }

    /// Set the CLIP weight file, in .safetensors format.
    pub fn with_clip_weights(mut self, clip_weights: impl Into<String>) -> Self {
        self.clip_weights = Some(clip_weights.into());
        self
    }

    /// Set the CLIP weight file used by the prior model, in .safetensors format.
    pub fn with_prior_clip_weights(mut self, prior_clip_weights: impl Into<String>) -> Self {
        self.prior_clip_weights = Some(prior_clip_weights.into());
        self
    }

    /// Set the prior weight file, in .safetensors format.
    pub fn with_prior_weights(mut self, prior_weights: impl Into<String>) -> Self {
        self.prior_weights = Some(prior_weights.into());
        self
    }

    /// Set the Vector Quantized Generative Adversarial Network weight file, in .safetensors format.
    pub fn with_vqgan_weights(mut self, vqgan_weights: impl Into<String>) -> Self {
        self.vqgan_weights = Some(vqgan_weights.into());
        self
    }

    /// Set the file specifying the tokenizer to used for tokenization.
    pub fn with_tokenizer(mut self, tokenizer: impl Into<String>) -> Self {
        self.tokenizer = Some(tokenizer.into());
        self
    }

    /// Set the file specifying the tokenizer to used for prior tokenization.
    pub fn with_prior_tokenizer(mut self, prior_tokenizer: impl Into<String>) -> Self {
        self.prior_tokenizer = Some(prior_tokenizer.into());
        self
    }

    /// Build the model.
    pub async fn build(self) -> Result<Wuerstchen, CacheError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    pub async fn build_with_loading_handler(
        self,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Wuerstchen, CacheError> {
        let WuerstchenBuilder {
            use_flash_attn,
            decoder_weights,
            clip_weights,
            prior_clip_weights,
            prior_weights,
            vqgan_weights,
            tokenizer,
            prior_tokenizer,
        } = self;

        // Download section
        let prior_tokenizer_source = ModelFile::PriorTokenizer.get(prior_tokenizer);
        let prior_tokenizer_source_display =
            format!("Prior Tokenizer ({})", prior_tokenizer_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(prior_tokenizer_source_display);
        let prior_tokenizer = prior_tokenizer_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let tokenizer_source = ModelFile::Tokenizer.get(tokenizer);
        let tokenizer_source_display = format!("Tokenizer ({})", tokenizer_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(tokenizer_source_display);
        let tokenizer = tokenizer_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let clip_weights_source = ModelFile::Clip.get(clip_weights);
        let clip_weights_source_display = format!("Clip Weights ({})", clip_weights_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(clip_weights_source_display);
        let clip_weights = clip_weights_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let prior_clip_weights_source = ModelFile::PriorClip.get(prior_clip_weights);
        let prior_clip_weights_source_display =
            format!("Prior Clip Weights ({})", prior_clip_weights_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(prior_clip_weights_source_display);
        let prior_clip_weights = prior_clip_weights_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let decoder_weights_source = ModelFile::Decoder.get(decoder_weights);
        let decoder_weights_source_display =
            format!("Decoder Weights ({})", decoder_weights_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(decoder_weights_source_display);
        let decoder_weights = decoder_weights_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let prior_weights_source = ModelFile::Prior.get(prior_weights);
        let prior_weights_source_display = format!("Prior Weights ({})", prior_weights_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(prior_weights_source_display);
        let prior_weights = prior_weights_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let vqgan_weights_source = ModelFile::VqGan.get(vqgan_weights);
        let vqgan_weights_source_display = format!("VQGAN Weights ({})", vqgan_weights_source);
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(vqgan_weights_source_display);
        let vqgan_weights = vqgan_weights_source
            .download(|progress| progress_handler(create_progress(progress)))
            .await?;

        let settings = WuerstcheModelSettings {
            use_flash_attn,
            decoder_weights,
            clip_weights,
            prior_clip_weights,
            prior_weights,
            vqgan_weights,
            tokenizer,
            prior_tokenizer,
        };
        let model = WuerstchenInner::new(settings).unwrap();

        let (rx, tx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async move {
                    while let Ok(message) = tx.recv() {
                        match message {
                            WuerstchenMessage::Kill => return,
                            WuerstchenMessage::Generate(input, result) => {
                                model.run(input, result);
                            }
                        }
                    }
                });
        });

        Ok(Wuerstchen {
            thread: Some(thread),
            sender: rx,
        })
    }
}

#[async_trait::async_trait]
impl ModelBuilder for WuerstchenBuilder {
    type Model = Wuerstchen;
    type Error = CacheError;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        let downloaded_decoder_weights = self.decoder_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Decoder).downloaded();
        let downloaded_clip_weights = self.clip_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Clip).downloaded();
        let downloaded_prior_clip_weights = self.prior_clip_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::PriorClip).downloaded();
        let downloaded_prior_weights = self.prior_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Prior).downloaded();
        let downloaded_vqgan_weights = self.vqgan_weights.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::VqGan).downloaded();
        let downloaded_tokenizer = self.tokenizer.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::Tokenizer).downloaded();
        let downloaded_prior_tokenizer = self.prior_tokenizer.is_none()
            || <&ModelFile as Into<FileSource>>::into(&ModelFile::PriorTokenizer).downloaded();

        !(downloaded_decoder_weights
            && downloaded_clip_weights
            && downloaded_prior_clip_weights
            && downloaded_prior_weights
            && downloaded_vqgan_weights
            && downloaded_tokenizer
            && downloaded_prior_tokenizer)
    }
}

/// A quantized wuerstchen image diffusion model
pub struct Wuerstchen {
    thread: Option<std::thread::JoinHandle<()>>,
    sender: std::sync::mpsc::Sender<WuerstchenMessage>,
}

impl Wuerstchen {
    /// Create a default Wuerstchen model.
    pub async fn new() -> Result<Self, CacheError> {
        Self::builder().build().await
    }

    /// Create a new builder for the Wuerstchen model.
    pub fn builder() -> WuerstchenBuilder {
        WuerstchenBuilder::default()
    }

    /// Run inference with the given settings.
    ///
    /// Dropping the returned channel will stop the inference early.
    pub fn run(&self, settings: WuerstchenInferenceSettings) -> ChannelImageStream<Image> {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        self.run_into(settings, sender);
        ChannelImageStream::from(receiver)
    }

    /// Run inference with the given settings into a stream of images
    ///
    /// Dropping the receiver will stop the inference early.
    pub fn run_into(
        &self,
        settings: WuerstchenInferenceSettings,
        sender: tokio::sync::mpsc::UnboundedSender<Image>,
    ) {
        _ = self
            .sender
            .send(WuerstchenMessage::Generate(settings, sender));
    }
}

impl Drop for Wuerstchen {
    fn drop(&mut self) {
        self.sender.send(WuerstchenMessage::Kill).unwrap();
        self.thread.take().unwrap().join().unwrap();
    }
}

enum WuerstchenMessage {
    Kill,
    Generate(
        WuerstchenInferenceSettings,
        tokio::sync::mpsc::UnboundedSender<Image>,
    ),
}

/// Settings for running inference with the Wuerstchen model.
pub struct WuerstchenInferenceSettings {
    /// The prompt to be used for image generation.
    prompt: String,

    uncond_prompt: String,

    /// The height in pixels of the generated image.
    height: usize,

    /// The width in pixels of the generated image.
    width: usize,

    /// The number of steps to run the inference for prior (stage C).
    prior_steps: usize,

    /// The number of steps to run the denoiser
    denoiser_steps: usize,

    /// The number of samples to generate.
    num_samples: i64,

    /// Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.
    prior_guidance_scale: f64,
}

impl WuerstchenInferenceSettings {
    /// Create a new settings object with the given prompt.
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),

            uncond_prompt: String::new(),

            height: 1024,

            width: 1024,

            prior_steps: 60,

            denoiser_steps: 12,

            num_samples: 1,

            prior_guidance_scale: 4.0,
        }
    }

    /// Set the negative prompt to be used for image generation.
    pub fn with_negative_prompt(mut self, uncond_prompt: impl Into<String>) -> Self {
        self.uncond_prompt = uncond_prompt.into();
        self
    }

    /// Set the height in pixels of the generated image.
    pub fn with_height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Set the width in pixels of the generated image.
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set the number of steps to run the prior for.
    pub fn with_prior_steps(mut self, prior_steps: usize) -> Self {
        self.prior_steps = prior_steps;
        self
    }

    /// Set the number of steps to run the denoiser for.
    pub fn with_denoiser_steps(mut self, denoiser_steps: usize) -> Self {
        self.denoiser_steps = denoiser_steps;
        self
    }

    /// Set the number of samples to generate.
    pub fn with_sample_count(mut self, sample_count: i64) -> Self {
        self.num_samples = sample_count;
        self
    }

    /// Set the prior guidance scale.
    pub fn with_prior_guidance_scale(mut self, prior_guidance_scale: f64) -> Self {
        self.prior_guidance_scale = prior_guidance_scale;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    PriorTokenizer,
    Clip,
    PriorClip,
    Decoder,
    VqGan,
    Prior,
}

impl ModelFile {
    fn get(&self, filename: Option<String>) -> FileSource {
        match filename {
            Some(filename) => FileSource::local(std::path::PathBuf::from(filename)),
            None => self.into(),
        }
    }
}

impl From<&ModelFile> for FileSource {
    fn from(val: &ModelFile) -> Self {
        let repo_main = "warp-ai/wuerstchen";
        let repo_prior = "warp-ai/wuerstchen-prior";
        let (repo, path) = match val {
            ModelFile::Tokenizer => (repo_main, "tokenizer/tokenizer.json"),
            ModelFile::PriorTokenizer => (repo_prior, "tokenizer/tokenizer.json"),
            ModelFile::Clip => (repo_main, "text_encoder/model.safetensors"),
            ModelFile::PriorClip => (repo_prior, "text_encoder/model.safetensors"),
            ModelFile::Decoder => (repo_main, "decoder/diffusion_pytorch_model.safetensors"),
            ModelFile::VqGan => (repo_main, "vqgan/diffusion_pytorch_model.safetensors"),
            ModelFile::Prior => (repo_prior, "prior/diffusion_pytorch_model.safetensors"),
        };
        FileSource::huggingface(repo.to_owned(), "main".to_owned(), path.to_owned())
    }
}

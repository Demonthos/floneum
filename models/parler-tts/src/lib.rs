//! A simple wrapper around the Parler speech-to-text models.
//!
//! Parler is a speech-to-text AI model that allows you to convert an input prompt into speech audio.
//! Contrary to many other speech-to-text models, Parler uses a description input to determine the speaker.
//!
//! **Audio Output Methods**
//! 
//! - Output as a raw PCM vec.
//! - Output as a `wav` file with the `wav` feature.
//! 
//! 
//! **Common Problems**
//! 
//! If the model does not output the full input text in speech, try increasing the temperature and decreasing the top-p setting. 
//! This comes with the tradeoff of potentially worse quality output and must be balanced to achieve the desired results.
//! Both settings can be found in [`GenerationSettings`].
//!
//! ### Example
//!
//! Basic usage of the `parler-tts` crate. There are a variety of configuration options not covered here -
//! namely [`GenerationSettings`].
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use parler_tts::{ParlerBuilder, ParlerSource};
//!
//! #[tokio::main]
//! async fn main() {
//!     let parler = ParlerBuilder::new()
//!         .with_source(ParlerSource::MiniV1)
//!         .build()
//!         .await
//!         .unwrap();
//!
//!     let task = parler.generate(
//!         "The quick brown fox jumps over the lazy dog.",
//!         "Will's voice is monotone yet slightly fast in delivery, with very clear audio.",
//!     );
//!
//!     let decoder = task.await.unwrap();
//!     let out = decoder.raw_pcm();
//!     println!("{out:?}");
//! }
//! ```
//!
#![warn(missing_docs)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use futures_channel::oneshot;
use kalosm_common::CacheError;
use std::{
    error::Error,
    sync::{mpsc, Arc},
};

mod builder;
mod model;
mod task;

pub use builder::{ParlerBuilder, ParlerSource};
pub use model::Decoder;
pub use task::{GenerationSeed, GenerationSettings, GenerationTask};

/// The Parler speech-to-text model.
#[derive(Clone)]
pub struct Parler {
    inner: Arc<ParlerDrop>,
}

impl Parler {
    /// Create a builder for a Parler model.
    pub fn builder() -> ParlerBuilder {
        ParlerBuilder::default()
    }

    /// Create a new default Parler model.
    pub async fn new() -> Result<Self, ParlerLoadingError> {
        let model = Self::builder().build().await?;
        Ok(model)
    }

    /// Generate speech from a prompt and description.
    ///
    /// This will return a [`GenerationTask`] which is a handle to a generation.
    pub fn generate(&self, prompt: impl ToString, description: impl ToString) -> GenerationTask {
        GenerationTask {
            settings: GenerationSettings::default(),
            prompt: prompt.to_string(),
            description: description.to_string(),
            sender: self.inner.sender.clone(),
            receiver: Default::default(),
        }
    }
}

struct ParlerDrop {
    thread: Option<std::thread::JoinHandle<()>>,
    sender: mpsc::Sender<ParlerMessage>,
}

impl Drop for ParlerDrop {
    fn drop(&mut self) {
        self.sender.send(ParlerMessage::Kill).unwrap();
        self.thread.take().unwrap().join().unwrap();
    }
}

enum ParlerMessage {
    Kill,
    Generate {
        settings: GenerationSettings,
        prompt: String,
        description: String,
        result: oneshot::Sender<Result<Decoder, ParlerError>>,
    },
}

/// An error that can occur when loading a [`Parler`] model.
#[derive(Debug, thiserror::Error)]
pub enum ParlerLoadingError {
    /// An error that can occur when trying to load a [`Parler`] model from huggingface or a local file.
    #[error("Failed to load model from huggingface or local file: {0}")]
    DownloadingError(#[from] CacheError),

    /// An error that can occur when trying to load a [`Parler`] model.
    #[error("Failed to parse model json: {0}")]
    ParseModel(Box<dyn Error>),

    /// An error that can occur when trying to load a [`Parler`] model.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] candle_core::Error),
    /// An error that can occur when trying to load the [`Parler`] tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when trying to load the [`Parler`] config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
}

/// An error that can occur when running a [`Parler`] model.
#[derive(Debug, thiserror::Error)]
pub enum ParlerError {
    /// An error that can occur when trying to run a [`Parler`] model.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// An error that can occur when encoding or decoding for a [`Parler`] model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(tokenizers::Error),
    
    /// An error that can occur when decoding Parler output into a wav file.
    #[cfg(feature = "wav")]
    #[error("Wav output error: {0}")]
    WavOutput(#[from] hound::Error),
}

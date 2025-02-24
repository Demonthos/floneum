use futures_channel::oneshot;
use std::{future::Future, sync::mpsc, task::Poll};

use crate::{Decoder, ParlerError, ParlerMessage};

/// A generation task.
///
/// A generation task is your handle to a single generate action for the model. Optionally provide
/// your own settings with the [`GenerationTask::with_settings`] method, and await this task for
/// the results.
pub struct GenerationTask {
    pub(crate) settings: GenerationSettings,
    pub(crate) prompt: String,
    pub(crate) description: String,
    pub(crate) sender: mpsc::Sender<ParlerMessage>,
    pub(crate) receiver: Option<oneshot::Receiver<Result<Decoder, ParlerError>>>,
}

impl GenerationTask {
    /// Provide custom settings for this generation task.
    pub fn with_settings(mut self, settings: GenerationSettings) -> Self {
        self.settings = settings;
        self
    }
}

impl Future for GenerationTask {
    type Output = Result<Decoder, ParlerError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let myself = self.get_mut();

        // Create the receiver and send start message
        if myself.receiver.is_none() {
            let (tx, rx) = oneshot::channel();
            myself.receiver = Some(rx);

            _ = myself.sender.send(ParlerMessage::Generate {
                settings: myself.settings.clone(),
                prompt: myself.prompt.clone(),
                description: myself.description.clone(),
                result: tx,
            });
        }

        let value = myself.receiver.as_mut().unwrap().try_recv().unwrap();
        match value {
            Some(val) => Poll::Ready(val),
            None => {
                cx.waker().clone().wake();
                Poll::Pending
            }
        }
    }
}

/// The seed to use for a generation.
///
/// The seed is used for deterministic output. If you want consistent results
/// for the same input, use the same seed. For more varied results, use a random seed.
#[derive(Debug, Clone, Copy)]
pub enum GenerationSeed {
    /// Provide your own seed.
    Provided(u64),
    /// Let us generate a random seed.
    Random,
}

/// Settings for a generation task.
/// 
/// These settings can be applied with the [`GenerationTask::with_setings`](GenerationTask::with_settings) method.
#[derive(Debug, Clone)]
pub struct GenerationSettings {
    seed: GenerationSeed,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_steps: usize,
}

impl GenerationSettings {
    /// Create new generation settings with the default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the value of the seed setting.
    pub fn seed(&self) -> GenerationSeed {
        self.seed
    }

    /// Set the value of the seed setting.
    ///
    /// The seed is used for deterministic output. If you want consistent results
    /// for the same input, use the same seed. For more varied results, use a random seed.
    pub fn with_seed(mut self, seed: GenerationSeed) -> Self {
        self.seed = seed;
        self
    }

    /// Get the value of the temperature setting.
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Set the value of the temperature setting.
    ///
    /// Temperature is how creative Parler will tend to be.
    /// Value must be at least 0.0
    pub fn with_temperature(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature.map(|val| val.max(0.0));
        self
    }

    /// Get the value of the top-p setting.
    pub fn top_p(&self) -> Option<f64> {
        self.top_p
    }

    /// Set the value of the top-p setting.
    ///
    /// Top-p, also known as nucleus sampling, is the cumulative probability threshold for selecting tokens.
    /// You can think about it as the threshold required for tokens to be selected by the model.
    pub fn with_top_p(mut self, top_p: Option<f64>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Get the value of the max steps setting.
    pub fn max_steps(&self) -> usize {
        self.max_steps
    }

    /// Set the value of the max steps setting.
    ///
    /// More steps will use more RAM or VRAM.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            seed: GenerationSeed::Random,
            temperature: Some(1.0),
            top_p: Some(0.1),
            max_steps: 512,
        }
    }
}

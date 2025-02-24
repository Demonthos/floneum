use crate::{GenerationSeed, GenerationSettings, ParlerError, ParlerLoadingError};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::parler_tts::{Config, Model},
};
use kalosm_common::accelerated_device_if_available;
use rand::{rngs::ThreadRng, Rng};
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub(crate) struct ParlerInner {
    device: Device,
    tokenizer: Tokenizer,
    model: Model,
    config: Config,
    rng: ThreadRng,
}

impl ParlerInner {
    pub(crate) fn new(
        model_files: Vec<PathBuf>,
        tokenizer_file: PathBuf,
        config_file: PathBuf,
    ) -> Result<Self, ParlerLoadingError> {
        let device = accelerated_device_if_available()?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_file)
            .map_err(ParlerLoadingError::LoadTokenizer)?;

        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_file).unwrap())
            .map_err(ParlerLoadingError::LoadConfig)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_files, DType::F32, &device)? };

        let model = Model::new(&config, vb)?;

        Ok(Self {
            device,
            tokenizer,
            model,
            config,
            rng: rand::rng(),
        })
    }

    pub(crate) fn generate(
        &mut self,
        settings: GenerationSettings,
        prompt: String,
        description: String,
    ) -> Result<Decoder, ParlerError> {
        // Serialize input into tensor tokens.
        let description_tokens = self
            .tokenizer
            .encode(description, true)
            .map_err(ParlerError::Tokenizer)?
            .get_ids()
            .to_vec();
        let description_tokens = Tensor::new(description_tokens, &self.device)?.unsqueeze(0)?;

        let prompt_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(ParlerError::Tokenizer)?
            .get_ids()
            .to_vec();
        let prompt_tokens = Tensor::new(prompt_tokens, &self.device)?.unsqueeze(0)?;

        // Generate
        let seed = match settings.seed() {
            GenerationSeed::Provided(val) => val,
            GenerationSeed::Random => self.rng.random(),
        };

        let logit_proc = LogitsProcessor::new(seed, settings.temperature(), settings.top_p());
        let codes = self.model.generate(
            &prompt_tokens,
            &description_tokens,
            logit_proc,
            settings.max_steps(),
        )?;

        // Decode
        let codes = codes.to_dtype(DType::I64)?;
        let codes = codes.unsqueeze(0)?;

        let codes = codes.to_device(&self.device)?;
        let pcm = self.model.audio_encoder.decode_codes(&codes)?;

        let pcm = pcm.i((0, 0))?;
        let pcm = pcm.to_vec1::<f32>()?;

        Ok(Decoder::new(pcm, self.config.audio_encoder.sampling_rate))
    }
}

/// A decoder for Parler output audio data.
pub struct Decoder {
    pcm: Vec<f32>,
    sample_rate: u32,
}

impl Decoder {
    /// Create a new decoder with the specified PCM data and sample rate.
    pub fn new(pcm: Vec<f32>, sample_rate: u32) -> Self {
        Self { pcm, sample_rate }
    }

    /// Get a clone of the raw PCM data outputted from the model.
    pub fn raw_pcm(&self) -> Vec<f32> {
        self.pcm.clone()
    }

    /// Get the sample rate of the PCM data.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[cfg(feature = "wav")]
    /// Output the PCM data into a wav file.
    pub fn to_wav(&self, path: impl AsRef<std::path::Path>) -> Result<(), ParlerError> {
        use hound::{SampleFormat, WavSpec, WavWriter};

        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = WavWriter::create(path, spec)?;
        for sample in self.pcm.iter() {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;

        Ok(())
    }
}

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{from_anyhow, MoshiError};
use moshi::mimi::{Config as MimiConfig, Mimi};
use moshi::tts;
use moshi::{candle, candle_nn};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Arc;

/// Opaque handle to a TTS (Text-to-Speech) instance
///
/// This structure holds all components needed for text-to-speech synthesis including
/// the T5 text encoder, language model, Mimi audio codec, and optional speaker conditioning.
pub struct MoshiTTS {
    model: tts::Model,
    mimi: Mimi,
    tokenizer: Arc<tokenizers::Tokenizer>,
    sample_rate: f64,
    cfg_alpha: f64,
}

/// Create a new TTS instance
///
/// Loads model files and creates a TTS instance for converting text to speech.
/// The instance includes a T5 encoder for processing text, a language model for generation,
/// and a Mimi codec for converting audio tokens to PCM. Optionally supports speaker
/// conditioning for voice cloning.
///
/// # Parameters
/// - `t5_model_path`: Null-terminated UTF-8 string path to the T5 encoder safetensors file
/// - `lm_model_path`: Null-terminated UTF-8 string path to the language model safetensors file
/// - `mimi_model_path`: Null-terminated UTF-8 string path to the Mimi codec safetensors file
/// - `speaker_model_path`: Null-terminated UTF-8 string path to speaker conditioning model (may be null)
/// - `tokenizer_path`: Null-terminated UTF-8 string path to the tokenizer JSON file
/// - `num_codebooks`: Number of audio codebooks (typically 8)
/// - `cfg_alpha`: Classifier-free guidance alpha (typically 3.0, higher = more faithful to prompt)
/// - `dtype`: Data type string, one of: "f32", "f16", "bf16" (null-terminated UTF-8)
/// - `error`: Output pointer for error code (may be null to ignore errors)
///
/// # Returns
/// - Non-null pointer to `MoshiTTS` instance on success
/// - Null pointer on failure (check `error` code and `moshi_last_error()` for details)
///
/// # Memory Ownership
/// The caller owns the returned pointer and must free it with `moshi_tts_free` when done.
/// The instance allocates substantial memory for model weights that will be freed when
/// `moshi_tts_free` is called.
///
/// # Safety
/// - All path parameters must be valid null-terminated UTF-8 strings (except speaker_model_path which may be null)
/// - Path pointers must remain valid for the duration of this call
/// - `error` must be null or point to valid writable memory
/// - The returned pointer must be freed exactly once with `moshi_tts_free`
/// - Do not use the returned pointer after freeing it
///
/// # Thread Safety
/// This function is thread-safe. Multiple threads can create separate TTS instances
/// concurrently. However, each returned instance is not thread-safe and should only be
/// used from a single thread.
#[no_mangle]
pub unsafe extern "C" fn moshi_tts_new(
    t5_model_path: *const c_char,
    lm_model_path: *const c_char,
    mimi_model_path: *const c_char,
    speaker_model_path: *const c_char,
    tokenizer_path: *const c_char,
    num_codebooks: usize,
    cfg_alpha: f64,
    dtype: *const c_char,
    error: *mut MoshiError,
) -> *mut MoshiTTS {
    if t5_model_path.is_null()
        || lm_model_path.is_null()
        || mimi_model_path.is_null()
        || tokenizer_path.is_null()
        || dtype.is_null()
    {
        if !error.is_null() {
            *error = MoshiError::NullPointer;
        }
        return std::ptr::null_mut();
    }

    let result = (|| {
        let t5_path = CStr::from_ptr(t5_model_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in T5 model path: {}", e))?;
        let lm_path = CStr::from_ptr(lm_model_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in LM model path: {}", e))?;
        let mimi_path = CStr::from_ptr(mimi_model_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in Mimi model path: {}", e))?;
        let tokenizer_path_str = CStr::from_ptr(tokenizer_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in tokenizer path: {}", e))?;
        let dtype_str = CStr::from_ptr(dtype)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in dtype: {}", e))?;

        let speaker_path = if speaker_model_path.is_null() {
            None
        } else {
            Some(
                CStr::from_ptr(speaker_model_path)
                    .to_str()
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in speaker model path: {}", e))?,
            )
        };

        let device = candle::Device::Cpu;
        let dtype = match dtype_str {
            "f32" => candle::DType::F32,
            "f16" => candle::DType::F16,
            "bf16" => candle::DType::BF16,
            _ => anyhow::bail!("Unsupported dtype '{}', must be one of: f32, f16, bf16", dtype_str),
        };

        // Load T5 encoder
        let t5_config = candle_transformers::models::t5::Config::config_t5_base();
        let vb_t5 = candle_nn::VarBuilder::from_mmaped_safetensors(
            &[PathBuf::from(t5_path)],
            dtype,
            &device,
        )?;

        // Load language model
        let vb_lm = candle_nn::VarBuilder::from_mmaped_safetensors(
            &[PathBuf::from(lm_path)],
            dtype,
            &device,
        )?;

        // Load optional speaker conditioning model
        let vb_speaker = match speaker_path {
            Some(path) => Some(candle_nn::VarBuilder::from_mmaped_safetensors(
                &[PathBuf::from(path)],
                dtype,
                &device,
            )?),
            None => None,
        };

        // Create TTS config and model
        let tts_config = if speaker_path.is_some() {
            tts::Config::v0_2(t5_config)
        } else {
            tts::Config::v0_1(t5_config)
        };

        let model = tts::Model::new(&tts_config, vb_t5, vb_lm, vb_speaker)?;

        // Load Mimi codec for audio generation
        let vb_mimi = candle_nn::VarBuilder::from_mmaped_safetensors(
            &[PathBuf::from(mimi_path)],
            candle::DType::F32,
            &device,
        )?;
        let mimi_cfg = MimiConfig::v0_1(Some(num_codebooks));
        let mimi = Mimi::new(mimi_cfg, vb_mimi)?;

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path_str)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Box::into_raw(Box::new(MoshiTTS {
            model,
            mimi,
            tokenizer: Arc::new(tokenizer),
            sample_rate: tts_config.mimi.sample_rate,
            cfg_alpha,
        })))
    })();

    match result {
        Ok(ptr) => {
            if !error.is_null() {
                *error = MoshiError::Ok;
            }
            ptr
        }
        Err(e) => {
            if !error.is_null() {
                *error = from_anyhow(e);
            }
            std::ptr::null_mut()
        }
    }
}

/// Free a TTS instance and all associated resources
///
/// Destroys the TTS instance and frees all memory including model weights, tokenizer,
/// and internal buffers. After calling this function, the pointer becomes invalid and
/// must not be used.
///
/// # Parameters
/// - `tts`: Pointer to a `MoshiTTS` instance created by `moshi_tts_new`, or null
///
/// # Safety
/// - `tts` must be null, or must have been created by `moshi_tts_new`
/// - `tts` must not have been previously freed
/// - After this call, `tts` becomes invalid and must not be dereferenced
/// - Any operations in progress on other threads using this instance will cause undefined behaviour
///
/// # Thread Safety
/// This function is not thread-safe with respect to the same handle. Do not free an instance
/// whilst other threads are actively using it.
#[no_mangle]
pub unsafe extern "C" fn moshi_tts_free(tts: *mut MoshiTTS) {
    if !tts.is_null() {
        let _ = Box::from_raw(tts);
    }
}

/// Synthesise speech from text
///
/// Converts text into PCM audio samples using the TTS model. The text is tokenized,
/// processed through the T5 encoder and language model, then decoded via the Mimi
/// codec to produce audio. Optionally accepts speaker audio for voice cloning.
///
/// # Parameters
/// - `tts`: Pointer to a `MoshiTTS` instance
/// - `text`: Null-terminated UTF-8 string containing the text to synthesise
/// - `speaker_pcm`: Optional speaker audio for voice conditioning (may be null)
/// - `speaker_samples`: Number of samples in speaker_pcm (ignored if speaker_pcm is null)
/// - `pcm_out`: Output pointer that will receive the allocated PCM audio array
/// - `samples_out`: Output pointer that will receive the number of audio samples
///
/// # Returns
/// - `MoshiError::Ok` (0) on success
/// - Non-zero error code on failure (retrieve message via `moshi_last_error`)
///
/// # Memory Ownership
/// On success, `*pcm_out` points to a newly allocated buffer containing the synthesised audio.
/// The caller owns this buffer and **must** free it with `moshi_free_buffer` when done.
/// The buffer remains valid until freed.
///
/// # Audio Format
/// - Sample rate: 24kHz (mono)
/// - Format: Float32 PCM
/// - Duration: Variable based on text length (max 60 seconds by default)
///
/// # Safety
/// - `tts` must be a valid pointer to a `MoshiTTS` instance
/// - `text` must be a valid null-terminated UTF-8 string
/// - If `speaker_pcm` is non-null, it must point to `speaker_samples` valid float32 values
/// - `pcm_out` must point to valid writable memory for a pointer
/// - `samples_out` must point to valid writable memory for a size_t
/// - All pointers must remain valid for the duration of this call
/// - The caller must free `*pcm_out` with `moshi_free_buffer` after use
///
/// # Thread Safety
/// Not thread-safe for the same `tts` instance. Do not call synthesise on the same
/// instance from multiple threads concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_tts_synthesise(
    tts: *mut MoshiTTS,
    text: *const c_char,
    speaker_pcm: *const f32,
    speaker_samples: usize,
    pcm_out: *mut *mut f32,
    samples_out: *mut usize,
) -> MoshiError {
    if tts.is_null() || text.is_null() || pcm_out.is_null() || samples_out.is_null() {
        return MoshiError::NullPointer;
    }

    let tts = &mut *tts;

    let result = (|| {
        let text_str = CStr::from_ptr(text)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in text: {}", e))?;

        // Tokenize text
        let encoding = tts
            .tokenizer
            .encode(text_str, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize text: {}", e))?;
        let token_ids = encoding.get_ids();

        // Convert token IDs to tensor
        let token_ids: Vec<u32> = token_ids.iter().map(|&id| id).collect();
        let dev = tts.model.lm.device();
        let token_tensor =
            candle::Tensor::from_vec(token_ids, (1, token_ids.len()), dev)?
                .to_dtype(candle::DType::U32)?;

        // Prepare speaker audio if provided
        let speaker_tensor = if !speaker_pcm.is_null() {
            let speaker_slice = std::slice::from_raw_parts(speaker_pcm, speaker_samples);
            Some(candle::Tensor::from_slice(
                speaker_slice,
                (1, 1, speaker_samples),
                dev,
            )?)
        } else {
            None
        };

        // Generate conditions
        let conditions = tts
            .model
            .conditions(&token_tensor, speaker_tensor.as_ref())?;

        // Sample audio tokens
        let audio_tokens = tts.model.sample(&conditions, tts.cfg_alpha)?;

        // Flatten audio tokens for Mimi decoding
        let steps = audio_tokens.len();
        if steps == 0 {
            anyhow::bail!("No audio tokens generated");
        }
        let codebooks = audio_tokens[0].len();

        let mut flat_tokens: Vec<u32> = Vec::with_capacity(steps * codebooks);
        for step in &audio_tokens {
            flat_tokens.extend_from_slice(step);
        }

        // Decode audio tokens to PCM
        let codes_tensor =
            candle::Tensor::from_vec(flat_tokens, (1, codebooks, steps), dev)?;
        let pcm_tensor = tts.mimi.decode(&codes_tensor)?;
        let pcm_vec = pcm_tensor.to_vec3::<f32>()?;

        // Flatten PCM output
        let mut flat_pcm: Vec<f32> = Vec::new();
        for batch in &pcm_vec {
            for channel in batch {
                flat_pcm.extend_from_slice(channel);
            }
        }

        *pcm_out = flat_pcm.as_mut_ptr();
        *samples_out = flat_pcm.len();
        std::mem::forget(flat_pcm); // Prevent deallocation

        Ok(())
    })();

    match result {
        Ok(()) => MoshiError::Ok,
        Err(e) => from_anyhow(e),
    }
}

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{from_anyhow, MoshiError};
use moshi::asr;
use moshi::lm::LmModel;
use moshi::mimi::{Config as MimiConfig, Mimi};
use moshi::{candle, candle_nn};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::Arc;

/// Opaque handle to an ASR (Automatic Speech Recognition) instance
///
/// This structure holds all components needed for speech recognition including the
/// language model, audio tokenizer, and text tokenizer.
pub struct MoshiASR {
    state: asr::State,
    text_tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
}

/// Create a new ASR instance
///
/// Loads model files and creates an ASR instance for converting speech to text.
/// The instance includes a language model for processing, an audio codec (Mimi) for
/// tokenizing audio, and a text tokenizer for converting tokens to readable text.
///
/// # Parameters
/// - `lm_model_path`: Null-terminated UTF-8 string path to the language model safetensors file
/// - `audio_tokenizer_path`: Null-terminated UTF-8 string path to the Mimi safetensors file
/// - `text_tokenizer_path`: Null-terminated UTF-8 string path to the SentencePiece model file
/// - `num_codebooks`: Number of audio codebooks (typically 8 or 16)
/// - `asr_delay_tokens`: ASR delay in tokens (typically 3-5 for lower latency)
/// - `temperature`: Sampling temperature (0.0 for greedy, higher for more randomness)
/// - `dtype`: Data type string, one of: "f32", "f16", "bf16" (null-terminated UTF-8)
/// - `error`: Output pointer for error code (may be null to ignore errors)
///
/// # Returns
/// - Non-null pointer to `MoshiASR` instance on success
/// - Null pointer on failure (check `error` code and `moshi_last_error()` for details)
///
/// # Memory Ownership
/// The caller owns the returned pointer and must free it with `moshi_asr_free` when done.
/// The instance allocates substantial memory for model weights that will be freed when
/// `moshi_asr_free` is called.
///
/// # Safety
/// - All path parameters must be valid null-terminated UTF-8 strings
/// - Path pointers must remain valid for the duration of this call
/// - `error` must be null or point to valid writable memory
/// - The returned pointer must be freed exactly once with `moshi_asr_free`
/// - Do not use the returned pointer after freeing it
///
/// # Thread Safety
/// This function is thread-safe. Multiple threads can create separate ASR instances
/// concurrently. However, each returned instance is not thread-safe and should only be
/// used from a single thread.
#[no_mangle]
pub unsafe extern "C" fn moshi_asr_new(
    lm_model_path: *const c_char,
    audio_tokenizer_path: *const c_char,
    text_tokenizer_path: *const c_char,
    num_codebooks: usize,
    asr_delay_tokens: usize,
    temperature: f64,
    dtype: *const c_char,
    error: *mut MoshiError,
) -> *mut MoshiASR {
    if lm_model_path.is_null()
        || audio_tokenizer_path.is_null()
        || text_tokenizer_path.is_null()
        || dtype.is_null()
    {
        if !error.is_null() {
            *error = MoshiError::NullPointer;
        }
        return std::ptr::null_mut();
    }

    let result = (|| {
        let lm_path = CStr::from_ptr(lm_model_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in LM model path: {}", e))?;
        let audio_path = CStr::from_ptr(audio_tokenizer_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in audio tokenizer path: {}", e))?;
        let text_path = CStr::from_ptr(text_tokenizer_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in text tokenizer path: {}", e))?;
        let dtype_str = CStr::from_ptr(dtype)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in dtype: {}", e))?;

        let device = candle::Device::Cpu;
        let dtype = match dtype_str {
            "f32" => candle::DType::F32,
            "f16" => candle::DType::F16,
            "bf16" => candle::DType::BF16,
            _ => anyhow::bail!("Unsupported dtype '{}', must be one of: f32, f16, bf16", dtype_str),
        };

        // Load language model
        let vb_lm = candle_nn::VarBuilder::from_mmaped_safetensors(
            &[PathBuf::from(lm_path)],
            dtype,
            &device,
        )?;

        // Create LM config for ASR
        let lm_config = moshi::lm::Config::asr_v0_1();
        let lm = LmModel::new(&lm_config, moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm))?;

        // Load audio tokenizer (Mimi)
        let vb_audio = candle_nn::VarBuilder::from_mmaped_safetensors(
            &[PathBuf::from(audio_path)],
            candle::DType::F32,
            &device,
        )?;
        let mut mimi_cfg = MimiConfig::v0_1(Some(num_codebooks));
        mimi_cfg.transformer.max_seq_len = lm_config.transformer.max_seq_len * 2;
        let audio_tokenizer = Mimi::new(mimi_cfg, vb_audio)?;

        // Load text tokenizer
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(text_path)
            .map_err(|e| anyhow::anyhow!("Failed to load text tokenizer: {}", e))?;

        // Create ASR state
        let state = asr::State::new(1, asr_delay_tokens, temperature, audio_tokenizer, lm)?;

        Ok(Box::into_raw(Box::new(MoshiASR {
            state,
            text_tokenizer: Arc::new(text_tokenizer),
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

/// Free an ASR instance and all associated resources
///
/// Destroys the ASR instance and frees all memory including model weights, tokenizers,
/// and internal buffers. After calling this function, the pointer becomes invalid and
/// must not be used.
///
/// # Parameters
/// - `asr`: Pointer to a `MoshiASR` instance created by `moshi_asr_new`, or null
///
/// # Safety
/// - `asr` must be null, or must have been created by `moshi_asr_new`
/// - `asr` must not have been previously freed
/// - After this call, `asr` becomes invalid and must not be dereferenced
/// - Any operations in progress on other threads using this instance will cause undefined behaviour
///
/// # Thread Safety
/// This function is not thread-safe with respect to the same handle. Do not free an instance
/// whilst other threads are actively using it.
#[no_mangle]
pub unsafe extern "C" fn moshi_asr_free(asr: *mut MoshiASR) {
    if !asr.is_null() {
        let _ = Box::from_raw(asr);
    }
}

/// Process PCM audio and return transcription as JSON
///
/// Converts raw PCM audio samples into text transcription. The audio is processed through
/// the audio tokenizer and language model to generate word-level transcriptions with timing
/// information. Results are returned as a JSON string.
///
/// # Parameters
/// - `asr`: Pointer to a `MoshiASR` instance
/// - `pcm`: Float32 PCM data array (24 kHz mono audio)
/// - `samples`: Number of audio samples (typically 1920 for 80ms frames at 24kHz)
/// - `result_json`: Output pointer that will receive the allocated JSON string
///
/// # Returns
/// - `MoshiError::Ok` (0) on success
/// - Non-zero error code on failure (retrieve message via `moshi_last_error`)
///
/// # Memory Ownership
/// On success, `*result_json` points to a newly allocated null-terminated string containing
/// the JSON result. The caller owns this string and **must** free it with `moshi_free_string`
/// when done. The string remains valid until freed.
///
/// # JSON Format
/// The returned JSON is an array of objects, each representing a transcription event:
///
/// ```json
/// [
///   {"type": "word", "text": "hello", "start_time": 0.5, "batch_idx": 0},
///   {"type": "end_word", "stop_time": 0.8, "batch_idx": 0}
/// ]
/// ```
///
/// # Safety
/// - `asr` must be a valid pointer to a `MoshiASR` instance
/// - `pcm` must point to `samples` valid float32 values
/// - `result_json` must point to valid writable memory for a pointer
/// - All pointers must remain valid for the duration of this call
/// - The caller must free `*result_json` with `moshi_free_string` after use
///
/// # Thread Safety
/// Not thread-safe for the same `asr` instance. Do not call transcribe on the same
/// instance from multiple threads concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_asr_transcribe(
    asr: *mut MoshiASR,
    pcm: *const f32,
    samples: usize,
    result_json: *mut *mut c_char,
) -> MoshiError {
    if asr.is_null() || pcm.is_null() || result_json.is_null() {
        return MoshiError::NullPointer;
    }

    let asr = &mut *asr;
    let pcm_slice = std::slice::from_raw_parts(pcm, samples);

    let result = (|| {
        let dev = asr.state.device();
        let pcm_tensor = candle::Tensor::from_slice(pcm_slice, (1, 1, samples), dev)?
            .to_dtype(candle::DType::F32)?;

        let messages = asr
            .state
            .step_pcm(pcm_tensor, None, &().into(), |_, _, _| ())?;

        // Convert messages to JSON
        let mut json_objects = Vec::new();
        for msg in messages {
            match msg {
                asr::AsrMsg::Word {
                    tokens,
                    start_time,
                    batch_idx,
                } => {
                    let text = asr
                        .text_tokenizer
                        .decode(&tokens)
                        .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
                    json_objects.push(serde_json::json!({
                        "type": "word",
                        "text": text,
                        "start_time": start_time,
                        "batch_idx": batch_idx
                    }));
                }
                asr::AsrMsg::EndWord {
                    stop_time,
                    batch_idx,
                } => {
                    json_objects.push(serde_json::json!({
                        "type": "end_word",
                        "stop_time": stop_time,
                        "batch_idx": batch_idx
                    }));
                }
                asr::AsrMsg::Step { .. } => {
                    // Skip step messages in the output
                }
            }
        }

        let json_str = serde_json::to_string(&json_objects)?;
        let c_str = CString::new(json_str)?;
        *result_json = c_str.into_raw();

        Ok(())
    })();

    match result {
        Ok(()) => MoshiError::Ok,
        Err(e) => from_anyhow(e),
    }
}

/// Reset the ASR state
///
/// Clears the internal state of the ASR instance, allowing it to start processing fresh
/// audio. This should be called between unrelated audio streams to prevent context from
/// one stream affecting another.
///
/// # Parameters
/// - `asr`: Pointer to a `MoshiASR` instance
///
/// # Returns
/// - `MoshiError::Ok` (0) on success
/// - Non-zero error code on failure (retrieve message via `moshi_last_error`)
///
/// # Safety
/// - `asr` must be a valid pointer to a `MoshiASR` instance
///
/// # Thread Safety
/// Not thread-safe for the same `asr` instance. Do not call reset whilst other operations
/// are in progress on the same instance.
#[no_mangle]
pub unsafe extern "C" fn moshi_asr_reset(asr: *mut MoshiASR) -> MoshiError {
    if asr.is_null() {
        return MoshiError::NullPointer;
    }

    let asr = &mut *asr;

    match asr.state.reset() {
        Ok(()) => MoshiError::Ok,
        Err(e) => from_anyhow(e),
    }
}

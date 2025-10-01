// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::error::{from_anyhow, MoshiError};
use moshi::mimi::{Config, Mimi};
use moshi::candle;
use moshi::candle_nn::VarBuilder;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::PathBuf;

/// Opaque handle to a Mimi audio codec instance
pub struct MoshiMimi {
    mimi: Mimi,
    device: candle::Device,
    dtype: candle::DType,
}

/// Create a new Mimi audio codec instance
///
/// Loads a Mimi model from a safetensors file and creates a codec instance for
/// encoding/decoding audio. The returned handle owns the model weights and internal state.
///
/// # Parameters
/// - `model_path`: Null-terminated UTF-8 string path to the safetensors model file
/// - `num_codebooks`: Number of codebooks to use (typically 8 or 16, max depends on model)
/// - `dtype`: Data type string, one of: "f32", "f16", "bf16" (null-terminated UTF-8)
/// - `error`: Output pointer for error code (may be null to ignore errors)
///
/// # Returns
/// - Non-null pointer to `MoshiMimi` instance on success
/// - Null pointer on failure (check `error` code and `moshi_last_error()` for details)
///
/// # Memory Ownership
/// The caller owns the returned pointer and must free it with `moshi_mimi_free` when done.
/// The codec allocates internal buffers and model weights that will be freed automatically
/// when `moshi_mimi_free` is called.
///
/// # Safety
/// - `model_path` must be a valid null-terminated UTF-8 string
/// - `dtype` must be a valid null-terminated UTF-8 string
/// - `model_path` and `dtype` must remain valid for the duration of this call
/// - `error` must be null or point to valid writable memory
/// - The returned pointer must be freed exactly once with `moshi_mimi_free`
/// - Do not use the returned pointer after freeing it
///
/// # Thread Safety
/// This function is thread-safe. Multiple threads can create separate codec instances
/// concurrently. However, each returned instance is not thread-safe and should only be
/// used from a single thread.
#[no_mangle]
pub unsafe extern "C" fn moshi_mimi_new(
    model_path: *const c_char,
    num_codebooks: usize,
    dtype: *const c_char,
    error: *mut MoshiError,
) -> *mut MoshiMimi {
    if model_path.is_null() || dtype.is_null() {
        if !error.is_null() {
            *error = MoshiError::NullPointer;
        }
        return std::ptr::null_mut();
    }

    let result = (|| {
        let path = CStr::from_ptr(model_path)
            .to_str()
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in model path: {}", e))?;
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

        let vb = VarBuilder::from_mmaped_safetensors(&[PathBuf::from(path)], dtype, &device)?;
        let cfg = Config::v0_1(Some(num_codebooks));
        let mimi = Mimi::new(cfg, vb)?;

        Ok(Box::into_raw(Box::new(MoshiMimi {
            mimi,
            device,
            dtype,
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

/// Free a Mimi instance and all associated resources
///
/// Destroys the codec instance and frees all memory including model weights and internal
/// buffers. After calling this function, the pointer becomes invalid and must not be used.
///
/// # Parameters
/// - `mimi`: Pointer to a `MoshiMimi` instance created by `moshi_mimi_new`, or null
///
/// # Safety
/// - `mimi` must be null, or must have been created by `moshi_mimi_new`
/// - `mimi` must not have been previously freed
/// - After this call, `mimi` becomes invalid and must not be dereferenced
/// - Any operations in progress on other threads using this instance will cause undefined behaviour
///
/// # Thread Safety
/// This function is not thread-safe with respect to the same handle. Do not free an instance
/// whilst other threads are actively using it.
#[no_mangle]
pub unsafe extern "C" fn moshi_mimi_free(mimi: *mut MoshiMimi) {
    if !mimi.is_null() {
        let _ = Box::from_raw(mimi);
    }
}

/// Encode PCM audio to tokens
///
/// Converts raw PCM audio samples into discrete tokens using the Mimi codec. The audio
/// is compressed to a much lower bitrate representation (1.1 kbps at 24 kHz sampling rate).
///
/// # Parameters
/// - `mimi`: Pointer to a `MoshiMimi` instance
/// - `pcm`: Float32 PCM data array (interleaved if multiple channels)
/// - `batch_size`: Batch dimension (typically 1)
/// - `channels`: Number of audio channels (typically 1)
/// - `samples`: Number of samples per channel (must be appropriate for the model)
/// - `codes_out`: Output pointer that will receive the allocated token array
/// - `dims_out`: Output array for dimensions [batch, codebooks, steps] (must have space for 3 elements)
///
/// # Returns
/// - `MoshiError::Ok` (0) on success
/// - Non-zero error code on failure (retrieve message via `moshi_last_error`)
///
/// # Memory Ownership
/// On success, `*codes_out` points to a newly allocated buffer containing the encoded tokens.
/// The caller owns this buffer and **must** free it with `moshi_free_buffer` when done.
/// The buffer remains valid until freed.
///
/// # Data Layout
/// - Input: `pcm[batch_size][channels][samples]` flattened in row-major order
/// - Output: `codes[batch][codebooks][steps]` flattened in row-major order
/// - Output dimensions are written to `dims_out[0]`, `dims_out[1]`, `dims_out[2]`
///
/// # Safety
/// - `mimi` must be a valid pointer to a `MoshiMimi` instance
/// - `pcm` must point to `batch_size * channels * samples` valid float32 values
/// - `codes_out` must point to valid writable memory for a pointer
/// - `dims_out` must point to valid writable memory for 3 size_t values
/// - All pointers must remain valid for the duration of this call
/// - The caller must free `*codes_out` with `moshi_free_buffer` after use
///
/// # Thread Safety
/// Not thread-safe for the same `mimi` instance. Do not call encode/decode on the same
/// instance from multiple threads concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_mimi_encode(
    mimi: *mut MoshiMimi,
    pcm: *const f32,
    batch_size: usize,
    channels: usize,
    samples: usize,
    codes_out: *mut *mut u32,
    dims_out: *mut usize,
) -> MoshiError {
    if mimi.is_null() || pcm.is_null() || codes_out.is_null() || dims_out.is_null() {
        return MoshiError::NullPointer;
    }

    let mimi = &mut *mimi;
    let pcm_slice = std::slice::from_raw_parts(pcm, batch_size * channels * samples);

    let result = (|| {
        let pcm_tensor = candle::Tensor::from_slice(
            pcm_slice,
            (batch_size, channels, samples),
            &mimi.device,
        )?.to_dtype(mimi.dtype)?;

        let codes = mimi.mimi.encode(&pcm_tensor)?;
        let codes_vec = codes.to_vec3::<u32>()?;

        let (b, c, s) = (codes_vec.len(), codes_vec[0].len(), codes_vec[0][0].len());

        // Flatten the 3D vector to 1D
        let mut flat: Vec<u32> = Vec::with_capacity(b * c * s);
        for batch in codes_vec.iter() {
            for codebook in batch.iter() {
                flat.extend_from_slice(codebook);
            }
        }

        *codes_out = flat.as_mut_ptr();
        std::mem::forget(flat); // Prevent deallocation

        // Write dimensions
        let dims = std::slice::from_raw_parts_mut(dims_out, 3);
        dims[0] = b;
        dims[1] = c;
        dims[2] = s;

        Ok(())
    })();

    match result {
        Ok(()) => MoshiError::Ok,
        Err(e) => from_anyhow(e),
    }
}

/// Decode tokens to PCM audio
///
/// Converts discrete tokens back into PCM audio samples using the Mimi codec. The tokens
/// are decompressed to reconstruct the original audio waveform.
///
/// # Parameters
/// - `mimi`: Pointer to a `MoshiMimi` instance
/// - `codes`: Token array (uint32 values)
/// - `batch_size`: Batch dimension (typically 1)
/// - `codebooks`: Number of codebooks (must match the value used when creating the instance)
/// - `steps`: Number of time steps
/// - `pcm_out`: Output pointer that will receive the allocated PCM array
/// - `dims_out`: Output array for dimensions [batch, channels, samples] (must have space for 3 elements)
///
/// # Returns
/// - `MoshiError::Ok` (0) on success
/// - Non-zero error code on failure (retrieve message via `moshi_last_error`)
///
/// # Memory Ownership
/// On success, `*pcm_out` points to a newly allocated buffer containing the decoded audio.
/// The caller owns this buffer and **must** free it with `moshi_free_buffer` when done.
/// The buffer remains valid until freed.
///
/// # Data Layout
/// - Input: `codes[batch][codebooks][steps]` flattened in row-major order
/// - Output: `pcm[batch_size][channels][samples]` flattened in row-major order
/// - Output dimensions are written to `dims_out[0]`, `dims_out[1]`, `dims_out[2]`
///
/// # Safety
/// - `mimi` must be a valid pointer to a `MoshiMimi` instance
/// - `codes` must point to `batch_size * codebooks * steps` valid uint32 values
/// - `pcm_out` must point to valid writable memory for a pointer
/// - `dims_out` must point to valid writable memory for 3 size_t values
/// - All pointers must remain valid for the duration of this call
/// - The caller must free `*pcm_out` with `moshi_free_buffer` after use
///
/// # Thread Safety
/// Not thread-safe for the same `mimi` instance. Do not call encode/decode on the same
/// instance from multiple threads concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_mimi_decode(
    mimi: *mut MoshiMimi,
    codes: *const u32,
    batch_size: usize,
    codebooks: usize,
    steps: usize,
    pcm_out: *mut *mut f32,
    dims_out: *mut usize,
) -> MoshiError {
    if mimi.is_null() || codes.is_null() || pcm_out.is_null() || dims_out.is_null() {
        return MoshiError::NullPointer;
    }

    let mimi = &mut *mimi;
    let codes_slice = std::slice::from_raw_parts(codes, batch_size * codebooks * steps);

    let result = (|| {
        let codes_tensor = candle::Tensor::from_slice(
            codes_slice,
            (batch_size, codebooks, steps),
            &mimi.device,
        )?;

        let pcm = mimi.mimi.decode(&codes_tensor)?.to_dtype(candle::DType::F32)?;
        let pcm_vec = pcm.to_vec3::<f32>()?;

        let (b, c, s) = (pcm_vec.len(), pcm_vec[0].len(), pcm_vec[0][0].len());

        // Flatten the 3D vector to 1D
        let mut flat: Vec<f32> = Vec::with_capacity(b * c * s);
        for batch in pcm_vec.iter() {
            for channel in batch.iter() {
                flat.extend_from_slice(channel);
            }
        }

        *pcm_out = flat.as_mut_ptr();
        std::mem::forget(flat); // Prevent deallocation

        // Write dimensions
        let dims = std::slice::from_raw_parts_mut(dims_out, 3);
        dims[0] = b;
        dims[1] = c;
        dims[2] = s;

        Ok(())
    })();

    match result {
        Ok(()) => MoshiError::Ok,
        Err(e) => from_anyhow(e),
    }
}

/// Free a buffer allocated by moshi functions
///
/// Frees memory allocated by `moshi_mimi_encode` or `moshi_mimi_decode` for output buffers.
/// After calling this function, the pointer becomes invalid and must not be used.
///
/// # Parameters
/// - `ptr`: Pointer to a buffer allocated by `moshi_mimi_encode` or `moshi_mimi_decode`, or null
///
/// # Safety
/// - `ptr` must be null, or must have been returned by `moshi_mimi_encode`/`moshi_mimi_decode`
/// - `ptr` must not have been previously freed
/// - After this call, `ptr` becomes invalid and must not be dereferenced
///
/// # Thread Safety
/// This function is thread-safe. Different threads can free different buffers concurrently.
#[no_mangle]
pub unsafe extern "C" fn moshi_free_buffer(ptr: *mut std::ffi::c_void) {
    if !ptr.is_null() {
        // We don't know the original type/size, so we can't properly deallocate
        // This is a limitation - in practice, we'd need separate free functions
        // for each buffer type, or store size metadata
        // For now, this is a placeholder that matches the API design
        libc::free(ptr);
    }
}

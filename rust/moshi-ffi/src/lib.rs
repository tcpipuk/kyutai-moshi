// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

//! C FFI bindings for the Moshi audio AI library
//!
//! This crate provides a C-compatible API for using Moshi's audio processing capabilities
//! (Mimi codec, ASR, and TTS) from other languages like Go, C++, or any language with C FFI support.
//!
//! # Architecture
//!
//! The FFI layer acts as a bridge between foreign language callers and the Rust core:
//!
//! ```text
//! Foreign Language (Go/C++) → C FFI (moshi-ffi) → Rust Core (moshi crate)
//! ```
//!
//! This design provides memory safety whilst maintaining performance for audio processing.
//! The FFI layer handles marshalling data between C representations and Rust types, manages
//! error propagation across the FFI boundary, and ensures proper resource cleanup.
//!
//! # Memory Management
//!
//! The FFI uses explicit manual memory management with clear ownership contracts:
//!
//! ## Opaque Handles
//!
//! Objects like `MoshiMimi*` are opaque pointers to Rust-owned heap allocations:
//! - Created by constructor functions (`moshi_mimi_new`)
//! - Must be freed with corresponding destructor (`moshi_mimi_free`)
//! - Pointers become invalid after freeing - do not use after free
//!
//! ## Output Buffers
//!
//! Functions that return audio or token data allocate buffers that the caller owns:
//! - `moshi_mimi_encode` returns token arrays via `codes_out` - free with `moshi_free_codes_buffer`
//! - `moshi_mimi_decode` returns PCM arrays via `pcm_out` - free with `moshi_free_pcm_buffer`
//! - `moshi_tts_synthesise` returns PCM arrays via `pcm_out` - free with `moshi_free_pcm_buffer`
//! - Caller must pass the correct buffer size when freeing
//! - Buffers remain valid until explicitly freed
//!
//! ## Error Strings
//!
//! Error messages are heap-allocated C strings:
//! - Retrieved via `moshi_last_error`
//! - Caller must free with `moshi_free_string`
//! - Cleared automatically on next error or via `moshi_clear_error`
//!
//! ## Example Lifecycle
//!
//! ```c
//! // Create codec (Rust allocates)
//! MoshiMimi* codec = moshi_mimi_new(path, 8, "f32", &error);
//!
//! // Encode audio (FFI allocates output buffer)
//! uint32_t* codes;
//! size_t dims[3];
//! moshi_mimi_encode(codec, pcm, 1, 1, 24000, &codes, dims);
//!
//! // Use the codes...
//! size_t size = dims[0] * dims[1] * dims[2];
//!
//! // Free output buffer (caller's responsibility)
//! moshi_free_codes_buffer(codes, size);
//!
//! // Free codec (caller's responsibility)
//! moshi_mimi_free(codec);
//! ```
//!
//! # Error Handling
//!
//! All fallible operations return a `MoshiError` code and store detailed messages:
//!
//! 1. Check the return code for errors
//! 2. If non-zero, retrieve the detailed message via `moshi_last_error`
//! 3. Free the error string with `moshi_free_string`
//!
//! ```c
//! MoshiError err = moshi_mimi_encode(...);
//! if (err != 0) {
//!     const char* msg = moshi_last_error();
//!     fprintf(stderr, "Encoding failed: %s\n", msg);
//!     moshi_free_string((char*)msg);
//!     return -1;
//! }
//! ```
//!
//! Error messages persist until the next error occurs or `moshi_clear_error` is called.
//!
//! # Thread Safety
//!
//! - **Opaque handles**: Not thread-safe. Each `MoshiMimi` instance should only be used
//!   from a single thread, or externally synchronized.
//! - **Error storage**: Uses a global mutex. Multiple threads can safely call functions
//!   concurrently, but error messages may be overwritten by other threads.
//! - **Recommendation**: Use separate codec instances per thread, and handle errors
//!   immediately after calls rather than deferring error retrieval.
//!
//! # Safety Requirements
//!
//! All functions marked as `unsafe` require the caller to ensure:
//!
//! - **Pointer validity**: All pointers must be non-null and properly aligned for their type
//! - **Lifetime**: Pointers must remain valid for the duration of the call
//! - **Freed pointers**: Never use pointers after freeing them
//! - **String encoding**: String pointers must be valid UTF-8 null-terminated C strings
//! - **Buffer sizes**: Input buffer sizes must match the actual data dimensions
//! - **Concurrent access**: Do not access the same handle from multiple threads

pub mod asr;
pub mod error;
pub mod mimi;
pub mod tts;

// Re-export public types and functions
pub use asr::{MoshiASR, moshi_asr_free, moshi_asr_new, moshi_asr_reset, moshi_asr_transcribe};
pub use error::{MoshiError, moshi_clear_error, moshi_free_string, moshi_last_error};
pub use mimi::{
    MoshiMimi, moshi_free_codes_buffer, moshi_free_pcm_buffer, moshi_mimi_decode,
    moshi_mimi_encode, moshi_mimi_free, moshi_mimi_new, moshi_mimi_reset,
};
pub use tts::{MoshiTTS, moshi_tts_free, moshi_tts_new, moshi_tts_reset, moshi_tts_synthesise};

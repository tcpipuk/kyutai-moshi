// Package moshi provides Go bindings for the Moshi audio AI library.
//
// Moshi is a speech-text foundation model and full-duplex spoken dialogue framework
// developed by Kyutai. This package exposes Moshi's core capabilities through a
// Go-friendly API using cgo to interface with the Rust implementation.
//
// # Architecture
//
// The bindings use a three-layer architecture:
//
//	Go API (moshi package) → C FFI (moshi-ffi crate) → Rust Core (moshi-core)
//
// This approach provides memory safety whilst maintaining performance for audio processing.
//
// # Components
//
// Currently implemented:
//   - Mimi: High-efficiency audio codec (24 kHz → 12.5 Hz, 1.1 kbps)
//   - ASR: Automatic speech recognition with word-level timing
//   - TTS: Text-to-speech synthesis with optional speaker conditioning
//
// # Memory Management
//
// Resources are managed through explicit Close() methods with finaliser backup:
//
//	codec, err := moshi.NewMimi(modelPath, config)
//	if err != nil {
//	    return err
//	}
//	defer codec.Close()  // Explicit cleanup (recommended)
//	// Finaliser runs automatically if Close() is forgotten
//
// Audio and token data is copied between C and Go for safety. Whilst this adds
// overhead compared to zero-copy approaches, it prevents memory corruption and
// use-after-free errors that are common in FFI code.
//
// # Error Handling
//
// All fallible operations return standard Go errors. Detailed error messages from
// the Rust layer are automatically retrieved and wrapped:
//
//	codes, shape, err := codec.Encode(pcm, audioShape)
//	if err != nil {
//	    log.Printf("Encoding failed: %v", err)
//	    return err
//	}
//
// # Performance Considerations
//
// For high-throughput applications:
//   - Batch multiple audio chunks when possible
//   - Use appropriate DType for your hardware (bf16 requires modern GPUs)
//   - Consider data layout to minimise copies
//
// # Example
//
//	package main
//
//	import (
//	    "log"
//	    "github.com/kyutai-labs/moshi/go/moshi"
//	)
//
//	func main() {
//	    // Create codec with default configuration
//	    config := moshi.DefaultMimiConfig()
//	    codec, err := moshi.NewMimi("/path/to/mimi.safetensors", config)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//	    defer codec.Close()
//
//	    // Encode 1 second of audio at 24kHz
//	    pcm := make([]float32, 24000)
//	    // ... fill with audio samples ...
//
//	    shape := moshi.AudioShape{
//	        Batch:    1,
//	        Channels: 1,
//	        Samples:  24000,
//	    }
//
//	    codes, codesShape, err := codec.Encode(pcm, shape)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // Decode back to audio
//	    decoded, _, err := codec.Decode(codes, codesShape)
//	    if err != nil {
//	        log.Fatal(err)
//	    }
//
//	    // decoded now contains reconstructed audio
//	    _ = decoded
//	}
//
// # Building
//
// This package requires the moshi-ffi C library to be built first. See the
// repository README for detailed build instructions.
//
// Quick build using Docker:
//
//	cd rust
//	docker run --rm -v $(pwd):/workspace -w /workspace rust:latest \
//	    cargo build -p moshi-ffi --release
//
// Then build the Go package:
//
//	cd go
//	export LD_LIBRARY_PATH=../rust/target/release:$LD_LIBRARY_PATH
//	go build ./moshi
package moshi

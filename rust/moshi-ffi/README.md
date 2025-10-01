# moshi-ffi

[![Crates.io](https://img.shields.io/crates/v/moshi-ffi.svg)](https://crates.io/crates/moshi-ffi)
[![Documentation](https://docs.rs/moshi-ffi/badge.svg)](https://docs.rs/moshi-ffi)

See the [top-level README.md](../../README.md) for more information on Moshi.

This provides C FFI bindings for Moshi, including **Mimi** (the streaming neural audio codec), **ASR** (automatic speech recognition), and **TTS** (text-to-speech synthesis). The C-compatible API enables use from Go, C++, and other languages.

See the [crate documentation](https://docs.rs/moshi-ffi) for the complete API reference.

## Building

Build using Docker if your system lacks a C compiler:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace/rust rust:latest \
  cargo build -p moshi-ffi --release
```

The resulting library will be in `target/release/libmoshi_ffi.so` (Linux), `libmoshi_ffi.dylib` (macOS), or `moshi_ffi.dll` (Windows). The C header is auto-generated at `rust/moshi-ffi/include/moshi.h`.

## Go bindings

See the `../../go/` directory for Go bindings using cgo.

## License

The present code is provided under the MIT and Apache 2.0 licenses.

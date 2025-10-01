# Moshi Go Bindings

[![Go Reference](https://pkg.go.dev/badge/github.com/kyutai-labs/moshi/go/moshi.svg)](https://pkg.go.dev/github.com/kyutai-labs/moshi/go/moshi)

See the [top-level README.md](../README.md) for more information on Moshi.

This provides Go bindings for Moshi via C FFI. The package provides idiomatic Go APIs for **Mimi** (the streaming neural audio codec), **ASR** (automatic speech recognition), and **TTS** (text-to-speech synthesis).

See the [package documentation](https://pkg.go.dev/github.com/kyutai-labs/moshi/go/moshi) for the complete API reference.

## Installation

```bash
go get github.com/kyutai-labs/moshi/go/moshi
```

Note: Requires the moshi-ffi C library. See `../rust/moshi-ffi/README.md` for build instructions.

## Quick Example

```go
package main

import (
    "log"
    "github.com/kyutai-labs/moshi/go/moshi"
)

func main() {
    codec, err := moshi.NewMimi("/path/to/mimi.safetensors", moshi.DefaultMimiConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer codec.Close()

    pcm := make([]float32, 24000)  // 1 second at 24kHz
    shape := moshi.AudioShape{Batch: 1, Channels: 1, Samples: 24000}

    codes, _, err := codec.Encode(pcm, shape)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Encoded %d tokens", len(codes))
}
```

See the `examples/` directory for more complete examples.

## License

The present code is provided under the MIT and Apache 2.0 licenses.

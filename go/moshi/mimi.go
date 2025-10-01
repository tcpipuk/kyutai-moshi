package moshi

/*
#cgo CFLAGS: -I../../rust/moshi-ffi/include
#cgo LDFLAGS: -L../../rust/target/release -lmoshi_ffi
#include <stdlib.h>
#include "moshi.h"
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Mimi represents a Mimi audio codec instance
type Mimi struct {
	ptr unsafe.Pointer
}

// MimiConfig holds configuration for creating a Mimi codec
type MimiConfig struct {
	NumCodebooks int
	DType        string // "f32", "f16", or "bf16"
}

// DefaultMimiConfig returns default configuration
func DefaultMimiConfig() MimiConfig {
	return MimiConfig{
		NumCodebooks: 8,
		DType:        "f32",
	}
}

// NewMimi creates a new Mimi audio codec instance
//
// Parameters:
//   - modelPath: Path to the safetensors model file
//   - config: Mimi configuration
//
// Returns:
//   - *Mimi instance or error
func NewMimi(modelPath string, config MimiConfig) (*Mimi, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	cDType := C.CString(config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.MoshiError
	ptr := C.moshi_mimi_new(cModelPath, C.ulong(config.NumCodebooks), cDType, &cerr)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("failed to create Mimi: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("failed to create Mimi: error code %d", cerr)
	}

	if ptr == nil {
		return nil, fmt.Errorf("failed to create Mimi: null pointer returned")
	}

	mimi := &Mimi{ptr: ptr}

	// Set up finaliser to free resources
	runtime.SetFinalizer(mimi, func(m *Mimi) {
		if m.ptr != nil {
			C.moshi_mimi_free(m.ptr)
			m.ptr = nil
		}
	})

	return mimi, nil
}

// Close frees the Mimi instance
// After calling Close, the instance should not be used
func (m *Mimi) Close() error {
	if m.ptr != nil {
		C.moshi_mimi_free(m.ptr)
		m.ptr = nil
		runtime.SetFinalizer(m, nil)
	}
	return nil
}

// AudioShape represents dimensions of audio data
type AudioShape struct {
	Batch    int
	Channels int
	Samples  int
}

// CodesShape represents dimensions of token data
type CodesShape struct {
	Batch     int
	Codebooks int
	Steps     int
}

// Encode converts PCM audio to tokens
//
// Parameters:
//   - pcm: Audio data as float32 slice [batch][channels][samples]
//   - shape: Dimensions of the audio data
//
// Returns:
//   - tokens as uint32 slice and shape, or error
func (m *Mimi) Encode(pcm []float32, shape AudioShape) ([]uint32, CodesShape, error) {
	if m.ptr == nil {
		return nil, CodesShape{}, fmt.Errorf("Mimi instance is closed")
	}

	expectedSize := shape.Batch * shape.Channels * shape.Samples
	if len(pcm) != expectedSize {
		return nil, CodesShape{}, fmt.Errorf("PCM size mismatch: got %d, expected %d", len(pcm), expectedSize)
	}

	var codesPtr *C.uint32_t
	dims := make([]C.ulong, 3)

	cerr := C.moshi_mimi_encode(
		m.ptr,
		(*C.float)(&pcm[0]),
		C.ulong(shape.Batch),
		C.ulong(shape.Channels),
		C.ulong(shape.Samples),
		&codesPtr,
		&dims[0],
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, CodesShape{}, fmt.Errorf("encode failed: %s", C.GoString(errMsg))
		}
		return nil, CodesShape{}, fmt.Errorf("encode failed: error code %d", cerr)
	}

	outShape := CodesShape{
		Batch:     int(dims[0]),
		Codebooks: int(dims[1]),
		Steps:     int(dims[2]),
	}

	// Copy data from C to Go
	size := outShape.Batch * outShape.Codebooks * outShape.Steps
	codes := make([]uint32, size)
	cSlice := (*[1 << 30]C.uint32_t)(unsafe.Pointer(codesPtr))[:size:size]
	for i := range codes {
		codes[i] = uint32(cSlice[i])
	}

	// Free the C-allocated buffer
	C.moshi_free_buffer(unsafe.Pointer(codesPtr))

	return codes, outShape, nil
}

// Decode converts tokens to PCM audio
//
// Parameters:
//   - codes: Token data as uint32 slice
//   - shape: Dimensions of the token data
//
// Returns:
//   - PCM audio as float32 slice and shape, or error
func (m *Mimi) Decode(codes []uint32, shape CodesShape) ([]float32, AudioShape, error) {
	if m.ptr == nil {
		return nil, AudioShape{}, fmt.Errorf("Mimi instance is closed")
	}

	expectedSize := shape.Batch * shape.Codebooks * shape.Steps
	if len(codes) != expectedSize {
		return nil, AudioShape{}, fmt.Errorf("codes size mismatch: got %d, expected %d", len(codes), expectedSize)
	}

	var pcmPtr *C.float
	dims := make([]C.ulong, 3)

	cerr := C.moshi_mimi_decode(
		m.ptr,
		(*C.uint32_t)(&codes[0]),
		C.ulong(shape.Batch),
		C.ulong(shape.Codebooks),
		C.ulong(shape.Steps),
		&pcmPtr,
		&dims[0],
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, AudioShape{}, fmt.Errorf("decode failed: %s", C.GoString(errMsg))
		}
		return nil, AudioShape{}, fmt.Errorf("decode failed: error code %d", cerr)
	}

	outShape := AudioShape{
		Batch:    int(dims[0]),
		Channels: int(dims[1]),
		Samples:  int(dims[2]),
	}

	// Copy data from C to Go
	size := outShape.Batch * outShape.Channels * outShape.Samples
	pcm := make([]float32, size)
	cSlice := (*[1 << 30]C.float)(unsafe.Pointer(pcmPtr))[:size:size]
	for i := range pcm {
		pcm[i] = float32(cSlice[i])
	}

	// Free the C-allocated buffer
	C.moshi_free_buffer(unsafe.Pointer(pcmPtr))

	return pcm, outShape, nil
}

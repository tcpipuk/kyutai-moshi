package moshi

/*
#cgo CFLAGS: -I../../rust/moshi-ffi/include
#cgo LDFLAGS: -L../../rust/target/release -lmoshi_ffi
#include <stdlib.h>
#include "moshi.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"runtime"
	"unsafe"
)

// ASR represents an Automatic Speech Recognition instance
type ASR struct {
	ptr unsafe.Pointer
}

// ASRConfig holds configuration for creating an ASR instance
type ASRConfig struct {
	NumCodebooks   int
	ASRDelayTokens int
	Temperature    float64
	DType          string // "f32", "f16", or "bf16"
}

// DefaultASRConfig returns default ASR configuration
func DefaultASRConfig() ASRConfig {
	return ASRConfig{
		NumCodebooks:   8,
		ASRDelayTokens: 3,
		Temperature:    0.0, // Greedy decoding
		DType:          "f32",
	}
}

// TranscriptionEvent represents a single event in the transcription output
type TranscriptionEvent struct {
	Type      string  `json:"type"` // "word" or "end_word"
	Text      string  `json:"text,omitempty"`
	StartTime float64 `json:"start_time,omitempty"`
	StopTime  float64 `json:"stop_time,omitempty"`
	BatchIdx  int     `json:"batch_idx,omitempty"`
}

// NewASR creates a new ASR instance
//
// This loads all necessary models including the language model, audio tokenizer (Mimi),
// and text tokenizer (SentencePiece). All three model files must be provided.
//
// Parameters:
//   - lmModelPath: Path to the language model safetensors file
//   - audioTokenizerPath: Path to the Mimi audio codec safetensors file
//   - textTokenizerPath: Path to the SentencePiece text tokenizer file
//   - config: ASR configuration
//
// Returns:
//   - *ASR instance or error
func NewASR(lmModelPath, audioTokenizerPath, textTokenizerPath string, config ASRConfig) (*ASR, error) {
	cLMPath := C.CString(lmModelPath)
	defer C.free(unsafe.Pointer(cLMPath))

	cAudioPath := C.CString(audioTokenizerPath)
	defer C.free(unsafe.Pointer(cAudioPath))

	cTextPath := C.CString(textTokenizerPath)
	defer C.free(unsafe.Pointer(cTextPath))

	cDType := C.CString(config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.MoshiError
	ptr := C.moshi_asr_new(
		cLMPath,
		cAudioPath,
		cTextPath,
		C.ulong(config.NumCodebooks),
		C.ulong(config.ASRDelayTokens),
		C.double(config.Temperature),
		cDType,
		&cerr,
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("failed to create ASR: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("failed to create ASR: error code %d", cerr)
	}

	if ptr == nil {
		return nil, fmt.Errorf("failed to create ASR: null pointer returned")
	}

	asr := &ASR{ptr: ptr}

	// Set up finaliser to free resources
	runtime.SetFinalizer(asr, func(a *ASR) {
		if a.ptr != nil {
			C.moshi_asr_free(a.ptr)
			a.ptr = nil
		}
	})

	return asr, nil
}

// Close frees the ASR instance
//
// After calling Close, the instance should not be used. This releases all model
// weights and internal state.
func (a *ASR) Close() error {
	if a.ptr != nil {
		C.moshi_asr_free(a.ptr)
		a.ptr = nil
		runtime.SetFinalizer(a, nil)
	}
	return nil
}

// Transcribe processes audio samples and returns transcription events
//
// The audio should be mono PCM at 24kHz. Typical frame sizes are 1920 samples
// (80ms at 24kHz). The function returns word-level transcriptions with timing
// information.
//
// Parameters:
//   - pcm: Audio samples as float32 slice
//
// Returns:
//   - Slice of transcription events or error
func (a *ASR) Transcribe(pcm []float32) ([]TranscriptionEvent, error) {
	if a.ptr == nil {
		return nil, fmt.Errorf("ASR instance is closed")
	}

	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data is empty")
	}

	var resultJSON *C.char
	cerr := C.moshi_asr_transcribe(
		a.ptr,
		(*C.float)(&pcm[0]),
		C.ulong(len(pcm)),
		&resultJSON,
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("transcription failed: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("transcription failed: error code %d", cerr)
	}

	// Convert JSON result to Go structures
	jsonStr := C.GoString(resultJSON)
	C.moshi_free_string(resultJSON)

	var events []TranscriptionEvent
	if err := json.Unmarshal([]byte(jsonStr), &events); err != nil {
		return nil, fmt.Errorf("failed to parse transcription result: %w", err)
	}

	return events, nil
}

// Reset clears the ASR internal state
//
// This should be called between unrelated audio streams to prevent context
// from one stream affecting another.
//
// Returns:
//   - error if reset fails
func (a *ASR) Reset() error {
	if a.ptr == nil {
		return fmt.Errorf("ASR instance is closed")
	}

	cerr := C.moshi_asr_reset(a.ptr)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return fmt.Errorf("reset failed: %s", C.GoString(errMsg))
		}
		return fmt.Errorf("reset failed: error code %d", cerr)
	}

	return nil
}

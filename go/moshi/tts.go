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

// TTS represents a Text-to-Speech instance
type TTS struct {
	ptr unsafe.Pointer
}

// TTSConfig holds configuration for creating a TTS instance
type TTSConfig struct {
	NumCodebooks int
	CFGAlpha     float64
	DType        string // "f32", "f16", or "bf16"
}

// DefaultTTSConfig returns default TTS configuration
func DefaultTTSConfig() TTSConfig {
	return TTSConfig{
		NumCodebooks: 8,
		CFGAlpha:     3.0, // Classifier-free guidance strength
		DType:        "f32",
	}
}

// NewTTS creates a new TTS instance
//
// This loads all necessary models including the T5 text encoder, language model,
// Mimi audio codec, and optionally a speaker conditioning model for voice cloning.
//
// Parameters:
//   - t5ModelPath: Path to the T5 encoder safetensors file
//   - lmModelPath: Path to the language model safetensors file
//   - mimiModelPath: Path to the Mimi audio codec safetensors file
//   - speakerModelPath: Path to speaker conditioning model (empty string for no conditioning)
//   - tokenizerPath: Path to the tokenizer JSON file
//   - config: TTS configuration
//
// Returns:
//   - *TTS instance or error
func NewTTS(t5ModelPath, lmModelPath, mimiModelPath, speakerModelPath, tokenizerPath string, config TTSConfig) (*TTS, error) {
	cT5Path := C.CString(t5ModelPath)
	defer C.free(unsafe.Pointer(cT5Path))

	cLMPath := C.CString(lmModelPath)
	defer C.free(unsafe.Pointer(cLMPath))

	cMimiPath := C.CString(mimiModelPath)
	defer C.free(unsafe.Pointer(cMimiPath))

	var cSpeakerPath *C.char
	if speakerModelPath != "" {
		cSpeakerPath = C.CString(speakerModelPath)
		defer C.free(unsafe.Pointer(cSpeakerPath))
	}

	cTokenizerPath := C.CString(tokenizerPath)
	defer C.free(unsafe.Pointer(cTokenizerPath))

	cDType := C.CString(config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.MoshiError
	ptr := C.moshi_tts_new(
		cT5Path,
		cLMPath,
		cMimiPath,
		cSpeakerPath,
		cTokenizerPath,
		C.ulong(config.NumCodebooks),
		C.double(config.CFGAlpha),
		cDType,
		&cerr,
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("failed to create TTS: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("failed to create TTS: error code %d", cerr)
	}

	if ptr == nil {
		return nil, fmt.Errorf("failed to create TTS: null pointer returned")
	}

	tts := &TTS{ptr: ptr}

	// Set up finaliser to free resources
	runtime.SetFinalizer(tts, func(t *TTS) {
		if t.ptr != nil {
			C.moshi_tts_free(t.ptr)
			t.ptr = nil
		}
	})

	return tts, nil
}

// Close frees the TTS instance
//
// After calling Close, the instance should not be used. This releases all model
// weights and internal state.
func (t *TTS) Close() error {
	if t.ptr != nil {
		C.moshi_tts_free(t.ptr)
		t.ptr = nil
		runtime.SetFinalizer(t, nil)
	}
	return nil
}

// Synthesise converts text to speech
//
// The text is tokenized, encoded, and converted to audio through the TTS model.
// Optionally accepts speaker audio samples for voice cloning (if the model was
// created with speaker conditioning support).
//
// Parameters:
//   - text: The text to convert to speech
//   - speakerAudio: Optional speaker audio for voice cloning (nil for default voice)
//
// Returns:
//   - PCM audio samples (24kHz mono) or error
func (t *TTS) Synthesise(text string, speakerAudio []float32) ([]float32, error) {
	if t.ptr == nil {
		return nil, fmt.Errorf("TTS instance is closed")
	}

	if text == "" {
		return nil, fmt.Errorf("text is empty")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var speakerPtr *C.float
	var speakerSamples C.ulong
	if len(speakerAudio) > 0 {
		speakerPtr = (*C.float)(&speakerAudio[0])
		speakerSamples = C.ulong(len(speakerAudio))
	}

	var pcmPtr *C.float
	var samplesOut C.ulong

	cerr := C.moshi_tts_synthesise(
		t.ptr,
		cText,
		speakerPtr,
		speakerSamples,
		&pcmPtr,
		&samplesOut,
	)

	if cerr != C.MoshiError(0) {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("synthesis failed: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("synthesis failed: error code %d", cerr)
	}

	// Copy PCM data from C to Go
	samples := int(samplesOut)
	pcm := make([]float32, samples)
	cSlice := (*[1 << 30]C.float)(unsafe.Pointer(pcmPtr))[:samples:samples]
	for i := range pcm {
		pcm[i] = float32(cSlice[i])
	}

	// Free the C-allocated buffer
	C.moshi_free_buffer(unsafe.Pointer(pcmPtr))

	return pcm, nil
}

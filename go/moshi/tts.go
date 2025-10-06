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
	"sync"
	"unsafe"
)

// TTS represents a Text-to-Speech instance.
//
// All instances use a pool architecture. By default, the pool has max=1 instance,
// providing the same behaviour as a single thread-safe instance. For concurrent
// processing, configure a larger pool size via TTSOption.
//
// Each pool instance contains a complete model (~1.5GB for TTS), allowing you to
// configure maximum concurrent streams based on your memory budget. For most home
// users handling 1-2 streams, the default max=1 is ideal.
type TTS struct {
	pool   *ttsPool
	mu     sync.RWMutex
	closed bool
}

// ttsPool manages a dynamic pool of TTS instances
type ttsPool struct {
	t5ModelPath      string
	lmModelPath      string
	mimiModelPath    string
	speakerModelPath string
	tokenizerPath    string
	config           TTSConfig

	mu        sync.Mutex
	instances []*ttsInstance
	minSize   int
	maxSize   int
	available chan *ttsInstance
	closed    bool
}

// ttsInstance wraps a single TTS instance within the pool
type ttsInstance struct {
	ptr    unsafe.Pointer
	inUse  bool
	pooled bool // true if part of a pool
}

// TTSConfig holds configuration for creating a TTS instance
type TTSConfig struct {
	NumCodebooks int
	CFGAlpha     float64 // Classifier-free guidance strength
	DType        string  // "f32" (CPU/all GPUs), "f16" (Pascal+ GPUs), "bf16" (Ampere+ GPUs only)
}

// TTSPoolConfig configures instance pooling behaviour
type TTSPoolConfig struct {
	MinInstances int // Minimum instances to maintain (default: 1)
	MaxInstances int // Maximum instances allowed (default: 1)
}

// TTSOption configures a TTS instance
type TTSOption func(*TTSPoolConfig)

// DefaultTTSConfig returns default TTS configuration
func DefaultTTSConfig() TTSConfig {
	return TTSConfig{
		NumCodebooks: 8,
		CFGAlpha:     3.0, // Classifier-free guidance strength
		DType:        "f32",
	}
}

// DefaultTTSPoolConfig returns default pool configuration
func DefaultTTSPoolConfig() TTSPoolConfig {
	return TTSPoolConfig{
		MinInstances: 1,
		MaxInstances: 1,
	}
}

// WithTTSPool configures pool size for concurrent processing
func WithTTSPool(min, max int) TTSOption {
	return func(cfg *TTSPoolConfig) {
		cfg.MinInstances = min
		cfg.MaxInstances = max
	}
}

// NewTTS creates a new TTS instance.
//
// By default, the instance uses a pool with max=1, providing thread-safe access
// via internal serialisation. For concurrent processing, use WithTTSPool option
// to configure a larger pool size.
//
// This loads all necessary models including the T5 text encoder, language model,
// Mimi audio codec, and optionally a speaker conditioning model for voice cloning.
//
// The pool grows dynamically from min to max instances under load. When all
// instances are busy, callers block until one becomes available.
//
// Memory per instance: ~1.5GB for TTS
// Example: WithTTSPool(1, 4) requires ~6GB memory for TTS instances
//
// Parameters:
//   - t5ModelPath: Path to the T5 encoder safetensors file
//   - lmModelPath: Path to the language model safetensors file
//   - mimiModelPath: Path to the Mimi audio codec safetensors file
//   - speakerModelPath: Path to speaker conditioning model (empty string for no conditioning)
//   - tokenizerPath: Path to the tokenizer JSON file
//   - config: TTS configuration
//   - opts: Optional configuration (e.g., WithTTSPool for concurrent processing)
//
// Returns:
//   - *TTS instance or error
//
// Example:
//
//	// Default: max=1 (thread-safe, serialised access)
//	tts, _ := NewTTS(t5Path, lmPath, mimiPath, "", tokPath, DefaultTTSConfig())
//
//	// Concurrent: max=4 (true parallelism)
//	tts, _ := NewTTS(t5Path, lmPath, mimiPath, "", tokPath, DefaultTTSConfig(), WithTTSPool(1, 4))
func NewTTS(t5ModelPath, lmModelPath, mimiModelPath, speakerModelPath, tokenizerPath string, config TTSConfig, opts ...TTSOption) (*TTS, error) {
	// Default: min=1, max=1 (backwards compatible behaviour)
	poolConfig := DefaultTTSPoolConfig()
	for _, opt := range opts {
		opt(&poolConfig)
	}

	if poolConfig.MinInstances < 1 {
		return nil, fmt.Errorf("minInstances must be at least 1")
	}
	if poolConfig.MaxInstances < poolConfig.MinInstances {
		return nil, fmt.Errorf("maxInstances (%d) must be >= minInstances (%d)",
			poolConfig.MaxInstances, poolConfig.MinInstances)
	}

	pool := &ttsPool{
		t5ModelPath:      t5ModelPath,
		lmModelPath:      lmModelPath,
		mimiModelPath:    mimiModelPath,
		speakerModelPath: speakerModelPath,
		tokenizerPath:    tokenizerPath,
		config:           config,
		minSize:          poolConfig.MinInstances,
		maxSize:          poolConfig.MaxInstances,
		available:        make(chan *ttsInstance, poolConfig.MaxInstances),
		instances:        make([]*ttsInstance, 0, poolConfig.MaxInstances),
	}

	// Create minimum instances
	for i := 0; i < poolConfig.MinInstances; i++ {
		inst, err := pool.createInstance()
		if err != nil {
			// Clean up any instances created so far
			pool.closeAll()
			return nil, fmt.Errorf("failed to create initial instance %d: %w", i, err)
		}
		pool.instances = append(pool.instances, inst)
		pool.available <- inst
	}

	tts := &TTS{pool: pool}

	// Set up finaliser to clean up pool
	runtime.SetFinalizer(tts, (*TTS).Close)

	return tts, nil
}

// createInstance creates a new TTS instance (called with pool.mu held)
func (p *ttsPool) createInstance() (*ttsInstance, error) {
	cT5Path := C.CString(p.t5ModelPath)
	defer C.free(unsafe.Pointer(cT5Path))

	cLMPath := C.CString(p.lmModelPath)
	defer C.free(unsafe.Pointer(cLMPath))

	cMimiPath := C.CString(p.mimiModelPath)
	defer C.free(unsafe.Pointer(cMimiPath))

	var cSpeakerPath *C.char
	if p.speakerModelPath != "" {
		cSpeakerPath = C.CString(p.speakerModelPath)
		defer C.free(unsafe.Pointer(cSpeakerPath))
	}

	cTokenizerPath := C.CString(p.tokenizerPath)
	defer C.free(unsafe.Pointer(cTokenizerPath))

	cDType := C.CString(p.config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.enum_MoshiError
	ptr := C.moshi_tts_new(
		cT5Path,
		cLMPath,
		cMimiPath,
		cSpeakerPath,
		cTokenizerPath,
		C.ulong(p.config.NumCodebooks),
		C.double(p.config.CFGAlpha),
		cDType,
		&cerr,
	)

	if cerr != C.MoshiError_Ok {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return nil, fmt.Errorf("failed to create instance: %s", C.GoString(errMsg))
		}
		return nil, fmt.Errorf("failed to create instance: error code %d", cerr)
	}

	if ptr == nil {
		return nil, fmt.Errorf("failed to create instance: null pointer returned")
	}

	return &ttsInstance{
		ptr:    unsafe.Pointer(ptr),
		pooled: true,
	}, nil
}

// acquire gets an instance from the pool, creating a new one if needed
func (p *ttsPool) acquire() (*ttsInstance, error) {
	// Try to get an available instance without blocking
	select {
	case inst := <-p.available:
		if inst.ptr == nil {
			return nil, fmt.Errorf("acquired instance with nil pointer")
		}
		inst.inUse = true
		return inst, nil
	default:
		// No available instances - try to create a new one if under max
		p.mu.Lock()
		defer p.mu.Unlock()

		if p.closed {
			return nil, fmt.Errorf("pool is closed")
		}

		// Check if we can create more instances
		if len(p.instances) < p.maxSize {
			inst, err := p.createInstance()
			if err != nil {
				return nil, err
			}
			p.instances = append(p.instances, inst)
			inst.inUse = true
			return inst, nil
		}
	}

	// Pool is at max size - block until an instance becomes available
	inst := <-p.available
	if inst.ptr == nil {
		return nil, fmt.Errorf("acquired instance with nil pointer")
	}
	inst.inUse = true
	return inst, nil
}

// release returns an instance to the pool after resetting its state
func (p *ttsPool) release(inst *ttsInstance) error {
	if inst == nil || inst.ptr == nil {
		return fmt.Errorf("cannot release nil instance")
	}

	// Reset instance state before returning to pool
	cerr := C.moshi_tts_reset((*C.struct_MoshiTTS)(inst.ptr))
	if cerr != C.MoshiError_Ok {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return fmt.Errorf("failed to reset instance: %s", C.GoString(errMsg))
		}
		return fmt.Errorf("failed to reset instance: error code %d", cerr)
	}

	inst.inUse = false
	p.available <- inst
	return nil
}

// closeAll shuts down the pool and frees all instances
func (p *ttsPool) closeAll() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return
	}
	p.closed = true

	// Close the available channel
	close(p.available)

	// Free all instances
	for _, inst := range p.instances {
		if inst.ptr != nil {
			C.moshi_tts_free((*C.struct_MoshiTTS)(inst.ptr))
			inst.ptr = nil
		}
	}
	p.instances = nil
}

// Close frees the TTS instance pool.
//
// After calling Close, the instance should not be used. This releases all model
// weights and internal state.
func (t *TTS) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.closed {
		return nil
	}

	runtime.SetFinalizer(t, nil)
	t.pool.closeAll()
	t.closed = true
	return nil
}

// Synthesise converts text to speech.
//
// This method is thread-safe. With default configuration (max=1), concurrent calls
// are serialised. With larger pool configuration, multiple calls can execute in
// parallel on independent TTS instances.
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
//   - PCM audio samples (24kHz mono, Float32 range -1.0 to 1.0) or error
func (t *TTS) Synthesise(text string, speakerAudio []float32) ([]float32, error) {
	if text == "" {
		return nil, fmt.Errorf("text is empty")
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		return nil, fmt.Errorf("TTS instance is closed")
	}

	inst, err := t.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer t.pool.release(inst)

	return t.synthesiseWithInstance(inst.ptr, text, speakerAudio)
}

// synthesiseWithInstance performs the actual synthesis with a specific instance pointer
func (t *TTS) synthesiseWithInstance(ptr unsafe.Pointer, text string, speakerAudio []float32) ([]float32, error) {
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
		(*C.struct_MoshiTTS)(ptr),
		cText,
		speakerPtr,
		speakerSamples,
		&pcmPtr,
		&samplesOut,
	)

	if cerr != C.MoshiError_Ok {
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

	// Free the C-allocated buffer using type-specific free function
	C.moshi_free_pcm_buffer(pcmPtr, C.ulong(samples))

	return pcm, nil
}

// Reset clears the TTS internal state.
//
// This should be called between unrelated synthesis requests to prevent context
// from one request affecting another.
//
// Note: Instances are automatically reset when released after Synthesise calls,
// so manual reset is typically not needed unless you want to reset mid-stream.
//
// Returns:
//   - error if reset fails
func (t *TTS) Reset() error {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if t.closed {
		return fmt.Errorf("TTS instance is closed")
	}

	// For explicit reset, acquire instance and reset it
	inst, err := t.pool.acquire()
	if err != nil {
		return fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer t.pool.release(inst)

	cerr := C.moshi_tts_reset((*C.struct_MoshiTTS)(inst.ptr))

	if cerr != C.MoshiError_Ok {
		errMsg := C.moshi_last_error()
		if errMsg != nil {
			defer C.moshi_free_string(errMsg)
			return fmt.Errorf("reset failed: %s", C.GoString(errMsg))
		}
		return fmt.Errorf("reset failed: error code %d", cerr)
	}

	return nil
}

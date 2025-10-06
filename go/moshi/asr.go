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
	"sync"
	"unsafe"
)

// ASR represents an Automatic Speech Recognition instance.
//
// All instances use a pool architecture. By default, the pool has max=1 instance,
// providing the same behaviour as a single thread-safe instance. For concurrent
// processing, configure a larger pool size via ASROption.
//
// Each pool instance contains a complete model (~2GB for ASR), allowing you to
// configure maximum concurrent streams based on your memory budget. For most home
// users handling 1-2 streams, the default max=1 is ideal.
type ASR struct {
	pool   *asrPool
	mu     sync.RWMutex
	closed bool
}

// asrPool manages a dynamic pool of ASR instances
type asrPool struct {
	lmModelPath        string
	audioTokenizerPath string
	textTokenizerPath  string
	config             ASRConfig

	mu        sync.Mutex
	instances []*asrInstance
	minSize   int
	maxSize   int
	available chan *asrInstance
	closed    bool
}

// asrInstance wraps a single ASR instance within the pool
type asrInstance struct {
	ptr    unsafe.Pointer
	inUse  bool
	pooled bool // true if part of a pool
}

// ASRConfig holds configuration for creating an ASR instance
type ASRConfig struct {
	NumCodebooks   int
	ASRDelayTokens int
	Temperature    float64
	DType          string // "f32" (CPU/all GPUs), "f16" (Pascal+ GPUs), "bf16" (Ampere+ GPUs only)
}

// ASRPoolConfig configures instance pooling behaviour
type ASRPoolConfig struct {
	MinInstances int // Minimum instances to maintain (default: 1)
	MaxInstances int // Maximum instances allowed (default: 1)
}

// ASROption configures an ASR instance
type ASROption func(*ASRPoolConfig)

// DefaultASRConfig returns default ASR configuration
func DefaultASRConfig() ASRConfig {
	return ASRConfig{
		NumCodebooks:   8,
		ASRDelayTokens: 3,
		Temperature:    0.0, // Greedy decoding
		DType:          "f32",
	}
}

// DefaultASRPoolConfig returns default pool configuration
func DefaultASRPoolConfig() ASRPoolConfig {
	return ASRPoolConfig{
		MinInstances: 1,
		MaxInstances: 1,
	}
}

// WithASRPool configures pool size for concurrent processing
func WithASRPool(min, max int) ASROption {
	return func(cfg *ASRPoolConfig) {
		cfg.MinInstances = min
		cfg.MaxInstances = max
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

// NewASR creates a new ASR instance.
//
// By default, the instance uses a pool with max=1, providing thread-safe access
// via internal serialisation. For concurrent processing, use WithASRPool option
// to configure a larger pool size.
//
// This loads all necessary models including the language model, audio tokenizer (Mimi),
// and text tokenizer (SentencePiece). All three model files must be provided.
//
// The pool grows dynamically from min to max instances under load. When all
// instances are busy, callers block until one becomes available.
//
// Memory per instance: ~2GB for ASR
// Example: WithASRPool(1, 4) requires ~8GB memory for ASR instances
//
// Parameters:
//   - lmModelPath: Path to the language model safetensors file
//   - audioTokenizerPath: Path to the Mimi audio codec safetensors file
//   - textTokenizerPath: Path to the SentencePiece text tokenizer file
//   - config: ASR configuration
//   - opts: Optional configuration (e.g., WithASRPool for concurrent processing)
//
// Returns:
//   - *ASR instance or error
//
// Example:
//
//	// Default: max=1 (thread-safe, serialised access)
//	asr, _ := NewASR(lmPath, audioPath, textPath, DefaultASRConfig())
//
//	// Concurrent: max=4 (true parallelism)
//	asr, _ := NewASR(lmPath, audioPath, textPath, DefaultASRConfig(), WithASRPool(1, 4))
func NewASR(lmModelPath, audioTokenizerPath, textTokenizerPath string, config ASRConfig, opts ...ASROption) (*ASR, error) {
	// Default: min=1, max=1 (backwards compatible behaviour)
	poolConfig := DefaultASRPoolConfig()
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

	pool := &asrPool{
		lmModelPath:        lmModelPath,
		audioTokenizerPath: audioTokenizerPath,
		textTokenizerPath:  textTokenizerPath,
		config:             config,
		minSize:            poolConfig.MinInstances,
		maxSize:            poolConfig.MaxInstances,
		available:          make(chan *asrInstance, poolConfig.MaxInstances),
		instances:          make([]*asrInstance, 0, poolConfig.MaxInstances),
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

	asr := &ASR{pool: pool}

	// Set up finaliser to clean up pool
	runtime.SetFinalizer(asr, (*ASR).Close)

	return asr, nil
}

// createInstance creates a new ASR instance (called with pool.mu held)
func (p *asrPool) createInstance() (*asrInstance, error) {
	cLMPath := C.CString(p.lmModelPath)
	defer C.free(unsafe.Pointer(cLMPath))

	cAudioPath := C.CString(p.audioTokenizerPath)
	defer C.free(unsafe.Pointer(cAudioPath))

	cTextPath := C.CString(p.textTokenizerPath)
	defer C.free(unsafe.Pointer(cTextPath))

	cDType := C.CString(p.config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.enum_MoshiError
	ptr := C.moshi_asr_new(
		cLMPath,
		cAudioPath,
		cTextPath,
		C.ulong(p.config.NumCodebooks),
		C.ulong(p.config.ASRDelayTokens),
		C.double(p.config.Temperature),
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

	return &asrInstance{
		ptr:    unsafe.Pointer(ptr),
		pooled: true,
	}, nil
}

// acquire gets an instance from the pool, creating a new one if needed
func (p *asrPool) acquire() (*asrInstance, error) {
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
func (p *asrPool) release(inst *asrInstance) error {
	if inst == nil || inst.ptr == nil {
		return fmt.Errorf("cannot release nil instance")
	}

	// Reset instance state before returning to pool
	cerr := C.moshi_asr_reset((*C.struct_MoshiASR)(inst.ptr))
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
func (p *asrPool) closeAll() {
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
			C.moshi_asr_free((*C.struct_MoshiASR)(inst.ptr))
			inst.ptr = nil
		}
	}
	p.instances = nil
}

// Close frees the ASR instance pool.
//
// After calling Close, the instance should not be used. This releases all model
// weights and internal state.
func (a *ASR) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.closed {
		return nil
	}

	runtime.SetFinalizer(a, nil)
	a.pool.closeAll()
	a.closed = true
	return nil
}

// Transcribe processes audio samples and returns transcription events.
//
// This method is thread-safe. With default configuration (max=1), concurrent calls
// are serialised. With larger pool configuration, multiple calls can execute in
// parallel on independent ASR instances.
//
// Audio format requirements:
//   - Sample rate: 24kHz (resample other rates first)
//   - Channels: Mono only (convert stereo to mono first)
//   - Format: Float32 PCM, range -1.0 to 1.0
//   - Frame size: Exact multiples of 1920 samples (80ms at 24kHz) for streaming
//   - No partial frames supported (GPU execution requires consistent frame sizes)
//
// The function returns word-level transcriptions with timing information.
//
// Parameters:
//   - pcm: Audio samples as float32 slice
//
// Returns:
//   - Slice of transcription events or error
func (a *ASR) Transcribe(pcm []float32) ([]TranscriptionEvent, error) {
	if len(pcm) == 0 {
		return nil, fmt.Errorf("PCM data is empty")
	}

	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.closed {
		return nil, fmt.Errorf("ASR instance is closed")
	}

	inst, err := a.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer a.pool.release(inst)

	return a.transcribeWithInstance(inst.ptr, pcm)
}

// transcribeWithInstance performs the actual transcription with a specific instance pointer
func (a *ASR) transcribeWithInstance(ptr unsafe.Pointer, pcm []float32) ([]TranscriptionEvent, error) {
	var resultJSON *C.char
	cerr := C.moshi_asr_transcribe(
		(*C.struct_MoshiASR)(ptr),
		(*C.float)(&pcm[0]),
		C.ulong(len(pcm)),
		&resultJSON,
	)

	if cerr != C.MoshiError_Ok {
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

// Reset clears the ASR internal state.
//
// This should be called between unrelated audio streams to prevent context
// from one stream affecting another.
//
// Note: Instances are automatically reset when released after Transcribe calls,
// so manual reset is typically not needed unless you want to reset mid-stream.
//
// Returns:
//   - error if reset fails
func (a *ASR) Reset() error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.closed {
		return fmt.Errorf("ASR instance is closed")
	}

	// For explicit reset, acquire instance and reset it
	inst, err := a.pool.acquire()
	if err != nil {
		return fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer a.pool.release(inst)

	cerr := C.moshi_asr_reset((*C.struct_MoshiASR)(inst.ptr))

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

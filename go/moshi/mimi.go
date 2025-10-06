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

// Mimi represents a Mimi audio codec instance.
//
// All instances use a pool architecture. By default, the pool has max=1 instance,
// providing the same behaviour as a single thread-safe instance. For concurrent
// processing, configure a larger pool size via MimiOption.
//
// Each pool instance contains a complete model (~250MB for Mimi), allowing you to
// configure maximum concurrent streams based on your memory budget. For most home
// users handling 1-2 streams, the default max=1 is ideal.
type Mimi struct {
	pool   *mimiPool
	mu     sync.RWMutex
	closed bool
}

// mimiPool manages a dynamic pool of Mimi instances
type mimiPool struct {
	modelPath string
	config    MimiConfig

	mu        sync.Mutex
	instances []*mimiInstance
	minSize   int
	maxSize   int
	available chan *mimiInstance
	closed    bool
}

// mimiInstance wraps a single codec instance within the pool
type mimiInstance struct {
	ptr    unsafe.Pointer
	inUse  bool
	pooled bool // true if part of a pool
}

// MimiConfig holds configuration for creating a Mimi codec
type MimiConfig struct {
	NumCodebooks int
	DType        string // "f32" (CPU/all GPUs), "f16" (Pascal+ GPUs), "bf16" (Ampere+ GPUs only)
}

// MimiPoolConfig configures instance pooling behaviour
type MimiPoolConfig struct {
	MinInstances int // Minimum instances to maintain (default: 1)
	MaxInstances int // Maximum instances allowed (default: 1)
}

// MimiOption configures a Mimi instance
type MimiOption func(*MimiPoolConfig)

// DefaultMimiConfig returns default configuration
func DefaultMimiConfig() MimiConfig {
	return MimiConfig{
		NumCodebooks: 8,
		DType:        "f32",
	}
}

// DefaultMimiPoolConfig returns default pool configuration
func DefaultMimiPoolConfig() MimiPoolConfig {
	return MimiPoolConfig{
		MinInstances: 1,
		MaxInstances: 1,
	}
}

// WithMimiPool configures pool size for concurrent processing
func WithMimiPool(min, max int) MimiOption {
	return func(cfg *MimiPoolConfig) {
		cfg.MinInstances = min
		cfg.MaxInstances = max
	}
}

// NewMimi creates a new Mimi audio codec instance.
//
// By default, the instance uses a pool with max=1, providing thread-safe access
// via internal serialisation. For concurrent processing, use WithMimiPool option
// to configure a larger pool size.
//
// The pool grows dynamically from min to max instances under load. When all
// instances are busy, callers block until one becomes available.
//
// Memory per instance: ~250MB for Mimi
// Example: WithMimiPool(1, 4) requires ~1GB memory for codec instances
//
// Parameters:
//   - modelPath: Path to the safetensors model file
//   - config: Mimi configuration
//   - opts: Optional configuration (e.g., WithMimiPool for concurrent processing)
//
// Returns:
//   - *Mimi instance or error
//
// Example:
//
//	// Default: max=1 (thread-safe, serialised access)
//	mimi, _ := NewMimi(modelPath, DefaultMimiConfig())
//
//	// Concurrent: max=4 (true parallelism)
//	mimi, _ := NewMimi(modelPath, DefaultMimiConfig(), WithMimiPool(1, 4))
func NewMimi(modelPath string, config MimiConfig, opts ...MimiOption) (*Mimi, error) {
	// Default: min=1, max=1 (backwards compatible behaviour)
	poolConfig := DefaultMimiPoolConfig()
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

	pool := &mimiPool{
		modelPath: modelPath,
		config:    config,
		minSize:   poolConfig.MinInstances,
		maxSize:   poolConfig.MaxInstances,
		available: make(chan *mimiInstance, poolConfig.MaxInstances),
		instances: make([]*mimiInstance, 0, poolConfig.MaxInstances),
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

	mimi := &Mimi{pool: pool}

	// Set up finaliser to clean up pool
	runtime.SetFinalizer(mimi, (*Mimi).Close)

	return mimi, nil
}

// createInstance creates a new codec instance (called with pool.mu held)
func (p *mimiPool) createInstance() (*mimiInstance, error) {
	cModelPath := C.CString(p.modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	cDType := C.CString(p.config.DType)
	defer C.free(unsafe.Pointer(cDType))

	var cerr C.enum_MoshiError
	ptr := C.moshi_mimi_new(cModelPath, C.ulong(p.config.NumCodebooks), cDType, &cerr)

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

	return &mimiInstance{
		ptr:    unsafe.Pointer(ptr),
		pooled: true,
	}, nil
}

// acquire gets an instance from the pool, creating a new one if needed
func (p *mimiPool) acquire() (*mimiInstance, error) {
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
func (p *mimiPool) release(inst *mimiInstance) error {
	if inst == nil || inst.ptr == nil {
		return fmt.Errorf("cannot release nil instance")
	}

	// Reset instance state before returning to pool
	cerr := C.moshi_mimi_reset((*C.struct_MoshiMimi)(inst.ptr))
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
func (p *mimiPool) closeAll() {
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
			C.moshi_mimi_free((*C.struct_MoshiMimi)(inst.ptr))
			inst.ptr = nil
		}
	}
	p.instances = nil
}

// Close frees the Mimi instance pool.
// After calling Close, the instance should not be used.
func (m *Mimi) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}

	runtime.SetFinalizer(m, nil)
	m.pool.closeAll()
	m.closed = true
	return nil
}

// AudioShape represents dimensions of audio data in batch-channel-sample format
type AudioShape struct {
	Batch    int // Batch dimension (typically 1 for single stream)
	Channels int // Number of audio channels (must be 1 for Mimi - mono only)
	Samples  int // Number of samples per channel at 24kHz (e.g., 24000 = 1 second)
}

// CodesShape represents dimensions of compressed token data in batch-codebook-step format
type CodesShape struct {
	Batch     int // Batch dimension (typically 1 for single stream)
	Codebooks int // Number of codebooks (8 or 16, must match model configuration)
	Steps     int // Number of time steps (24kHz → 12.5Hz: steps = samples/192)
}

// Encode converts PCM audio to tokens.
//
// This method is thread-safe. With default configuration (max=1), concurrent calls
// are serialised. With larger pool configuration, multiple calls can execute in
// parallel on independent codec instances.
//
// Parameters:
//   - pcm: Audio data as float32 slice [batch][channels][samples]
//   - shape: Dimensions of the audio data
//
// Returns:
//   - tokens as uint32 slice and shape, or error
func (m *Mimi) Encode(pcm []float32, shape AudioShape) ([]uint32, CodesShape, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, CodesShape{}, fmt.Errorf("Mimi instance is closed")
	}

	inst, err := m.pool.acquire()
	if err != nil {
		return nil, CodesShape{}, fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer m.pool.release(inst)

	return m.encodeWithInstance(inst.ptr, pcm, shape)
}

// encodeWithInstance performs the actual encoding with a specific instance pointer
func (m *Mimi) encodeWithInstance(ptr unsafe.Pointer, pcm []float32, shape AudioShape) ([]uint32, CodesShape, error) {
	expectedSize := shape.Batch * shape.Channels * shape.Samples
	if len(pcm) != expectedSize {
		return nil, CodesShape{}, fmt.Errorf("PCM size mismatch: got %d, expected %d", len(pcm), expectedSize)
	}

	var codesPtr *C.uint32_t
	dims := make([]C.ulong, 3)

	cerr := C.moshi_mimi_encode(
		(*C.struct_MoshiMimi)(ptr),
		(*C.float)(&pcm[0]),
		C.ulong(shape.Batch),
		C.ulong(shape.Channels),
		C.ulong(shape.Samples),
		&codesPtr,
		&dims[0],
	)

	if cerr != C.MoshiError_Ok {
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

	// Free the C-allocated buffer using type-specific free function
	C.moshi_free_codes_buffer(codesPtr, C.ulong(size))

	return codes, outShape, nil
}

// Decode converts tokens to PCM audio.
//
// This method is thread-safe. With default configuration (max=1), concurrent calls
// are serialised. With larger pool configuration, multiple calls can execute in
// parallel on independent codec instances.
//
// Parameters:
//   - codes: Token data as uint32 slice
//   - shape: Dimensions of the token data
//
// Returns:
//   - PCM audio as float32 slice and shape, or error
func (m *Mimi) Decode(codes []uint32, shape CodesShape) ([]float32, AudioShape, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return nil, AudioShape{}, fmt.Errorf("Mimi instance is closed")
	}

	inst, err := m.pool.acquire()
	if err != nil {
		return nil, AudioShape{}, fmt.Errorf("failed to acquire instance: %w", err)
	}
	defer m.pool.release(inst)

	return m.decodeWithInstance(inst.ptr, codes, shape)
}

// decodeWithInstance performs the actual decoding with a specific instance pointer
func (m *Mimi) decodeWithInstance(ptr unsafe.Pointer, codes []uint32, shape CodesShape) ([]float32, AudioShape, error) {
	expectedSize := shape.Batch * shape.Codebooks * shape.Steps
	if len(codes) != expectedSize {
		return nil, AudioShape{}, fmt.Errorf("codes size mismatch: got %d, expected %d", len(codes), expectedSize)
	}

	var pcmPtr *C.float
	dims := make([]C.ulong, 3)

	cerr := C.moshi_mimi_decode(
		(*C.struct_MoshiMimi)(ptr),
		(*C.uint32_t)(&codes[0]),
		C.ulong(shape.Batch),
		C.ulong(shape.Codebooks),
		C.ulong(shape.Steps),
		&pcmPtr,
		&dims[0],
	)

	if cerr != C.MoshiError_Ok {
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

	// Free the C-allocated buffer using type-specific free function
	C.moshi_free_pcm_buffer(pcmPtr, C.ulong(size))

	return pcm, outShape, nil
}

package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/kyutai-labs/moshi/go/moshi"
)

func main() {
	modelPath := flag.String("model", "", "Path to Mimi safetensors model")
	codebooks := flag.Int("codebooks", 8, "Number of codebooks")
	dtype := flag.String("dtype", "f32", "Data type (f32, f16, bf16)")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model is required")
		flag.Usage()
		os.Exit(1)
	}

	// Create Mimi codec
	config := moshi.MimiConfig{
		NumCodebooks: *codebooks,
		DType:        *dtype,
	}

	fmt.Printf("Loading Mimi model from %s...\n", *modelPath)
	codec, err := moshi.NewMimi(*modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create Mimi: %v", err)
	}
	defer codec.Close()

	fmt.Println("Model loaded successfully!")

	// Generate test audio: 1 second of 440 Hz sine wave at 24kHz
	sampleRate := 24000
	duration := 1.0
	frequency := 440.0
	samples := int(float64(sampleRate) * duration)

	pcm := make([]float32, samples)
	for i := range pcm {
		t := float64(i) / float64(sampleRate)
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*frequency*t))
	}

	fmt.Printf("Generated test audio: %d samples at %d Hz\n", len(pcm), sampleRate)

	// Encode
	fmt.Println("Encoding audio to tokens...")
	audioShape := moshi.AudioShape{
		Batch:    1,
		Channels: 1,
		Samples:  samples,
	}

	codes, codesShape, err := codec.Encode(pcm, audioShape)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}

	fmt.Printf("Encoded to %d tokens (shape: %d x %d x %d)\n",
		len(codes), codesShape.Batch, codesShape.Codebooks, codesShape.Steps)

	// Decode
	fmt.Println("Decoding tokens back to audio...")
	decoded, decodedShape, err := codec.Decode(codes, codesShape)
	if err != nil {
		log.Fatalf("Failed to decode: %v", err)
	}

	fmt.Printf("Decoded to %d samples (shape: %d x %d x %d)\n",
		len(decoded), decodedShape.Batch, decodedShape.Channels, decodedShape.Samples)

	// Calculate reconstruction error
	if len(decoded) == len(pcm) {
		var mse float64
		for i := range pcm {
			diff := float64(pcm[i] - decoded[i])
			mse += diff * diff
		}
		mse /= float64(len(pcm))
		fmt.Printf("Mean squared error: %.6f\n", mse)
		fmt.Printf("PSNR: %.2f dB\n", 10*math.Log10(1.0/mse))
	}

	fmt.Println("\nSuccess! Encode/decode cycle completed.")
}

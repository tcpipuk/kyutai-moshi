package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"

	"github.com/kyutai-labs/moshi/go/moshi"
)

func main() {
	lmModel := flag.String("lm-model", "", "Path to language model safetensors")
	audioModel := flag.String("audio-model", "", "Path to Mimi audio codec safetensors")
	textModel := flag.String("text-model", "", "Path to SentencePiece text tokenizer")
	codebooks := flag.Int("codebooks", 8, "Number of codebooks")
	delayTokens := flag.Int("delay", 3, "ASR delay in tokens")
	temperature := flag.Float64("temperature", 0.0, "Sampling temperature")
	dtype := flag.String("dtype", "f32", "Data type (f32, f16, bf16)")
	flag.Parse()

	if *lmModel == "" || *audioModel == "" || *textModel == "" {
		fmt.Fprintln(os.Stderr, "Error: --lm-model, --audio-model, and --text-model are required")
		flag.Usage()
		os.Exit(1)
	}

	// Create ASR instance
	config := moshi.ASRConfig{
		NumCodebooks:   *codebooks,
		ASRDelayTokens: *delayTokens,
		Temperature:    *temperature,
		DType:          *dtype,
	}

	fmt.Printf("Loading ASR models...\n")
	fmt.Printf("  LM Model: %s\n", *lmModel)
	fmt.Printf("  Audio Model: %s\n", *audioModel)
	fmt.Printf("  Text Model: %s\n", *textModel)

	asr, err := moshi.NewASR(*lmModel, *audioModel, *textModel, config)
	if err != nil {
		log.Fatalf("Failed to create ASR: %v", err)
	}
	defer asr.Close()

	fmt.Println("Models loaded successfully!")

	// Generate test audio: 80ms of 440 Hz sine wave at 24kHz (typical frame size)
	sampleRate := 24000
	frameDuration := 0.08 // 80ms
	frequency := 440.0
	samples := int(float64(sampleRate) * frameDuration)

	pcm := make([]float32, samples)
	for i := range pcm {
		t := float64(i) / float64(sampleRate)
		pcm[i] = float32(0.5 * math.Sin(2*math.Pi*frequency*t))
	}

	fmt.Printf("\nGenerated test audio: %d samples (%.0fms at %d Hz)\n",
		len(pcm), frameDuration*1000, sampleRate)

	// Process audio frames
	fmt.Println("\nTranscribing audio...")
	events, err := asr.Transcribe(pcm)
	if err != nil {
		log.Fatalf("Failed to transcribe: %v", err)
	}

	if len(events) == 0 {
		fmt.Println("No transcription events (this is expected for synthetic audio)")
	} else {
		fmt.Printf("Received %d transcription events:\n", len(events))
		var currentWords []string
		for _, event := range events {
			switch event.Type {
			case "word":
				fmt.Printf("  [%.2fs] Word: %q\n", event.StartTime, event.Text)
				currentWords = append(currentWords, event.Text)
			case "end_word":
				fmt.Printf("  [%.2fs] End of word\n", event.StopTime)
			}
		}

		if len(currentWords) > 0 {
			fmt.Printf("\nComplete transcription: %s\n", strings.Join(currentWords, " "))
		}
	}

	fmt.Println("\nSuccess! ASR processing completed.")
	fmt.Println("\nNote: For real speech transcription, provide actual audio samples.")
	fmt.Println("Audio should be mono PCM at 24kHz, typically in 80ms frames (1920 samples).")
}

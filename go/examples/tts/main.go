package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/kyutai-labs/moshi/go/moshi"
)

func main() {
	t5Model := flag.String("t5-model", "", "Path to T5 encoder safetensors")
	lmModel := flag.String("lm-model", "", "Path to language model safetensors")
	mimiModel := flag.String("mimi-model", "", "Path to Mimi codec safetensors")
	speakerModel := flag.String("speaker-model", "", "Path to speaker conditioning model (optional)")
	tokenizer := flag.String("tokenizer", "", "Path to tokenizer JSON file")
	text := flag.String("text", "Hello, this is a test of text to speech.", "Text to synthesise")
	codebooks := flag.Int("codebooks", 8, "Number of codebooks")
	cfgAlpha := flag.Float64("cfg-alpha", 3.0, "Classifier-free guidance strength")
	dtype := flag.String("dtype", "f32", "Data type (f32, f16, bf16)")
	flag.Parse()

	if *t5Model == "" || *lmModel == "" || *mimiModel == "" || *tokenizer == "" {
		fmt.Fprintln(os.Stderr, "Error: --t5-model, --lm-model, --mimi-model, and --tokenizer are required")
		fmt.Fprintln(os.Stderr, "\nRecommended test models:")
		fmt.Fprintln(os.Stderr, "  TTS Model: https://huggingface.co/kyutai/tts-0.75b-en-public")
		fmt.Fprintln(os.Stderr, "  Voice (optional): https://huggingface.co/kyutai/tts-voices/resolve/main/ears/p078/freeform_speech_01.wav.1e68beda%40240.safetensors")
		flag.Usage()
		os.Exit(1)
	}

	// Create TTS instance
	config := moshi.TTSConfig{
		NumCodebooks: *codebooks,
		CFGAlpha:     *cfgAlpha,
		DType:        *dtype,
	}

	fmt.Printf("Loading TTS models...\n")
	fmt.Printf("  T5 Model: %s\n", *t5Model)
	fmt.Printf("  LM Model: %s\n", *lmModel)
	fmt.Printf("  Mimi Model: %s\n", *mimiModel)
	if *speakerModel != "" {
		fmt.Printf("  Speaker Model: %s\n", *speakerModel)
	}
	fmt.Printf("  Tokenizer: %s\n", *tokenizer)

	tts, err := moshi.NewTTS(*t5Model, *lmModel, *mimiModel, *speakerModel, *tokenizer, config)
	if err != nil {
		log.Fatalf("Failed to create TTS: %v", err)
	}
	defer tts.Close()

	fmt.Println("Models loaded successfully!")

	// Synthesise speech
	fmt.Printf("\nSynthesising text: %q\n", *text)
	pcm, err := tts.Synthesise(*text, nil)
	if err != nil {
		log.Fatalf("Failed to synthesise: %v", err)
	}

	sampleRate := 24000
	duration := float64(len(pcm)) / float64(sampleRate)

	fmt.Printf("\nSuccess! Generated %.2f seconds of audio (%d samples at %d Hz)\n",
		duration, len(pcm), sampleRate)

	fmt.Println("\nTo save as WAV file, you can use a library like:")
	fmt.Println("  github.com/go-audio/wav")
	fmt.Println("\nRecommended test models:")
	fmt.Println("  TTS Model: https://huggingface.co/kyutai/tts-0.75b-en-public")
	fmt.Println("  Voice: https://huggingface.co/kyutai/tts-voices/resolve/main/ears/p078/freeform_speech_01.wav.1e68beda%40240.safetensors")
	fmt.Println("\nFor ASR testing:")
	fmt.Println("  STT Model: https://huggingface.co/kyutai/stt-1b-en_fr-candle (includes VAD)")
}

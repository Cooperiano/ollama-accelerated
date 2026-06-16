// Mixed sequence length benchmark - Tests PagedAttention continuous batching advantage
// This is where PagedAttention should shine with variable-length request scheduling
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

const (
	baseURL     = "http://localhost:11434"
	model       = "mistral:latest"
)

type GenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type GenerateResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// Prompts of varying lengths
var prompts = struct {
	short []string
	medium []string
	long []string
}{
	short: []string{
		"What is AI?",
		"Explain ML",
		"Define NN",
		"What is GPU?",
		"Explain CNN",
	},
	medium: []string{
		"Explain the concept of machine learning in detail. Cover supervised learning, unsupervised learning, and reinforcement learning with examples.",
		"Describe how neural networks work. Include information about layers, activation functions, backpropagation, and training algorithms.",
		"What is deep learning? Explain its relationship to neural networks, mention key architectures like CNNs and RNNs, and discuss applications.",
		"Explain natural language processing. Cover tokenization, embeddings, transformers, and modern approaches like GPT and BERT.",
		"Describe computer vision tasks. Include image classification, object detection, segmentation, and mention key techniques.",
	},
	long: []string{
		"Provide a comprehensive explanation of transformer architecture in deep learning. Start with the motivation behind transformers, explain the self-attention mechanism in detail, discuss multi-head attention, positional encodings, encoder-decoder structure, and how transformers revolutionized NLP. Include comparisons with RNNs and CNNs, mention scaling laws, and discuss famous transformer models like GPT, BERT, and T5. Conclude with recent developments and future directions.",
		"Explain the complete machine learning pipeline from data collection to deployment. Cover data preprocessing and feature engineering, model selection and training, validation and testing techniques, hyperparameter tuning strategies, deployment considerations, monitoring and maintenance. Include discussion of different ML paradigms, common challenges, and best practices. Use specific examples and mention popular tools and frameworks.",
		"Describe reinforcement learning in depth. Explain the core concepts of agents, environments, states, actions, and rewards. Cover value-based methods like Q-learning, policy-based methods, actor-critic architectures, and modern approaches like PPO. Discuss exploration vs exploitation, the credit assignment problem, and applications in robotics, games, and beyond. Include examples like AlphaGo and recent advances.",
		"Provide a detailed overview of computer vision techniques. Start from traditional methods like edge detection and feature extraction, move through the CNN revolution with architectures like LeNet, AlexNet, VGG, and ResNet. Cover object detection (YOLO, R-CNN), segmentation (U-Net, Mask R-CNN), and modern vision transformers. Discuss applications, challenges, and current state-of-the-art approaches.",
		"Explain natural language processing from foundations to state-of-the-art. Cover linguistic foundations, statistical methods, the neural network revolution, word embeddings, sequence models (RNNs, LSTMs), the transformer breakthrough, large language models, prompting and fine-tuning, multimodal models, and ethical considerations. Include key papers, models, and practical applications.",
	},
}

type RequestType struct {
	name       string
	prompts    []string
	weight     int
	maxTokens  int
}

func countTokens(s string) int {
	words := 0
	inWord := false
	for _, c := range s {
		if c == ' ' || c == '\n' || c == '\t' {
			if inWord {
				words++
				inWord = false
			}
		} else {
			inWord = true
		}
	}
	if inWord {
		words++
	}
	return words
}

func generateRequest(prompt string) (*GenerateResponse, time.Duration, int, error) {
	req := GenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
	}

	body, _ := json.Marshal(req)
	start := time.Now()

	resp, err := http.Post(baseURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, 0, 0, err
	}
	defer resp.Body.Close()

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, 0, 0, err
	}

	latency := time.Since(start)
	tokens := countTokens(result.Response)
	return &result, latency, tokens, nil
}

func warmup() {
	fmt.Print("  Warming up with mixed lengths...")
	shortPrompt := prompts.short[0]
	mediumPrompt := prompts.medium[0]
	longPrompt := prompts.long[0]

	generateRequest(shortPrompt)
	generateRequest(mediumPrompt)
	generateRequest(longPrompt)
	fmt.Println(" done")
}

type MixedLengthResult struct {
	Label              string
	TotalRequests      int
	SuccessCount       int32
	FailureCount       int32
	TotalTime          time.Duration
	// By type
	ShortCount         int
	ShortAvgTokens     float64
	ShortAvgLatency    time.Duration
	MediumCount        int
	MediumAvgTokens    float64
	MediumAvgLatency   time.Duration
	LongCount          int
	LongAvgTokens      float64
	LongAvgLatency     time.Duration
	// Overall
	OverallThroughput  float64
}

func runMixedLengthTest(label string, concurrency int, totalRequests int) *MixedLengthResult {
	fmt.Printf("\n%s=== Mixed Length Test: %s (Concurrency: %d, Requests: %d) ===%s\n",
		"\033[0;34m", label, concurrency, totalRequests, "\033[0m")

	warmup()

	// Define request types
	types := []RequestType{
		{name: "short", prompts: prompts.short, weight: 3, maxTokens: 100},
		{name: "medium", prompts: prompts.medium, weight: 2, maxTokens: 300},
		{name: "long", prompts: prompts.long, weight: 1, maxTokens: 800},
	}

	semaphore := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	var successCount atomic.Int32
	var failureCount atomic.Int32

	// Per-type metrics
	var shortTokens atomic.Int64
	var shortLatency atomic.Int64
	var shortCount atomic.Int32

	var mediumTokens atomic.Int64
	var mediumLatency atomic.Int64
	var mediumCount atomic.Int32

	var longTokens atomic.Int64
	var longLatency atomic.Int64
	var longCount atomic.Int32

	start := time.Now()

	// Generate mixed workload
	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(reqID int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			// Select type based on weights
			typeIdx := reqID % 6  // 0-2: short, 3-4: medium, 5: long
			var reqType RequestType
			var promptIdx int

			if typeIdx < 3 {
				reqType = types[0]  // short
				promptIdx = reqID % len(prompts.short)
			} else if typeIdx < 5 {
				reqType = types[1]  // medium
				promptIdx = reqID % len(prompts.medium)
			} else {
				reqType = types[2]  // long
				promptIdx = reqID % len(prompts.long)
			}

			prompt := reqType.prompts[promptIdx]
			_, latency, tokens, err := generateRequest(prompt)

			if err != nil {
				failureCount.Add(1)
				return
			}

			successCount.Add(1)

			// Record metrics by type
			latencyNs := latency.Nanoseconds()
			switch reqType.name {
			case "short":
				shortTokens.Add(int64(tokens))
				shortLatency.Add(latencyNs)
				shortCount.Add(1)
			case "medium":
				mediumTokens.Add(int64(tokens))
				mediumLatency.Add(latencyNs)
				mediumCount.Add(1)
			case "long":
				longTokens.Add(int64(tokens))
				longLatency.Add(latencyNs)
				longCount.Add(1)
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(start)

	result := &MixedLengthResult{
		Label:         label,
		TotalRequests: totalRequests,
		SuccessCount:  successCount.Load(),
		FailureCount:  failureCount.Load(),
		TotalTime:     totalTime,
	}

	// Calculate per-type averages
	if sc := shortCount.Load(); sc > 0 {
		result.ShortCount = int(sc)
		result.ShortAvgTokens = float64(shortTokens.Load()) / float64(sc)
		result.ShortAvgLatency = time.Duration(shortLatency.Load() / int64(sc))
	}

	if mc := mediumCount.Load(); mc > 0 {
		result.MediumCount = int(mc)
		result.MediumAvgTokens = float64(mediumTokens.Load()) / float64(mc)
		result.MediumAvgLatency = time.Duration(mediumLatency.Load() / int64(mc))
	}

	if lc := longCount.Load(); lc > 0 {
		result.LongCount = int(lc)
		result.LongAvgTokens = float64(longTokens.Load()) / float64(lc)
		result.LongAvgLatency = time.Duration(longLatency.Load() / int64(lc))
	}

	// Overall throughput
	totalTokens := shortTokens.Load() + mediumTokens.Load() + longTokens.Load()
	result.OverallThroughput = float64(totalTokens) / totalTime.Seconds()

	// Print results
	fmt.Printf("\n  Distribution: %d short, %d medium, %d long\n",
		result.ShortCount, result.MediumCount, result.LongCount)
	fmt.Printf("  Total time: %v\n", totalTime)
	fmt.Printf("  Success: %d, Failed: %d\n", result.SuccessCount, result.FailureCount)

	fmt.Printf("\n  By Type:\n")
	fmt.Printf("    Short:   %d req, %.1f tok, %v avg, %.1f tok/s\n",
		result.ShortCount, result.ShortAvgTokens, result.ShortAvgLatency,
		float64(result.ShortAvgTokens)/result.ShortAvgLatency.Seconds())
	fmt.Printf("    Medium:  %d req, %.1f tok, %v avg, %.1f tok/s\n",
		result.MediumCount, result.MediumAvgTokens, result.MediumAvgLatency,
		float64(result.MediumAvgTokens)/result.MediumAvgLatency.Seconds())
	fmt.Printf("    Long:    %d req, %.1f tok, %v avg, %.1f tok/s\n",
		result.LongCount, result.LongAvgTokens, result.LongAvgLatency,
		float64(result.LongAvgTokens)/result.LongAvgLatency.Seconds())

	fmt.Printf("\n  Overall Throughput: %.2f tok/s\n", result.OverallThroughput)

	return result
}

func printMixedLengthComparison(off, paged *MixedLengthResult) {
	fmt.Printf("\n%s=== Mixed Length Comparison ===%s\n", "\033[0;34m", "\033[0m")

	fmt.Println("┌─────────────────────┬──────────────────────┬──────────────────────┬──────────────┐")
	fmt.Printf("│ %-19s │ %-20s │ %-20s │ %-12s │\n", "Metric", "Official", "PagedAttn", "Diff")
	fmt.Println("├─────────────────────┼──────────────────────┼──────────────────────┼──────────────┤")

	fmt.Printf("│ %-19s │ %-20s │ %-20s │ %-12s │\n", "Total Time", fmt.Sprintf("%v", off.TotalTime), fmt.Sprintf("%v", paged.TotalTime), "")

	offTPS := off.OverallThroughput
	pagedTPS := paged.OverallThroughput
	improvement := ((pagedTPS - offTPS) / offTPS) * 100

	fmt.Printf("│ %-19s │ %-20.2f │ %-20.2f │ %+11.1f%% │\n", "Overall TPS", offTPS, pagedTPS, improvement)

	fmt.Println("├─────────────────────┼──────────────────────┼──────────────────────┼──────────────┤")

	fmt.Printf("│ %-19s │ %5.1f tok @ %5.1f tok/s │ %5.1f tok @ %5.1f tok/s │ %+11.1f%% │\n",
		"Short", off.ShortAvgTokens, float64(off.ShortAvgTokens)/off.ShortAvgLatency.Seconds(),
		paged.ShortAvgTokens, float64(paged.ShortAvgTokens)/paged.ShortAvgLatency.Seconds(),
		((float64(paged.ShortAvgTokens)/paged.ShortAvgLatency.Seconds())-(float64(off.ShortAvgTokens)/off.ShortAvgLatency.Seconds()))/(float64(off.ShortAvgTokens)/off.ShortAvgLatency.Seconds())*100)

	fmt.Printf("│ %-19s │ %5.1f tok @ %5.1f tok/s │ %5.1f tok @ %5.1f tok/s │ %+11.1f%% │\n",
		"Medium", off.MediumAvgTokens, float64(off.MediumAvgTokens)/off.MediumAvgLatency.Seconds(),
		paged.MediumAvgTokens, float64(paged.MediumAvgTokens)/paged.MediumAvgLatency.Seconds(),
		((float64(paged.MediumAvgTokens)/paged.MediumAvgLatency.Seconds())-(float64(off.MediumAvgTokens)/off.MediumAvgLatency.Seconds()))/(float64(off.MediumAvgTokens)/off.MediumAvgLatency.Seconds())*100)

	fmt.Printf("│ %-19s │ %5.1f tok @ %5.1f tok/s │ %5.1f tok @ %5.1f tok/s │ %+11.1f%% │\n",
		"Long", off.LongAvgTokens, float64(off.LongAvgTokens)/off.LongAvgLatency.Seconds(),
		paged.LongAvgTokens, float64(paged.LongAvgTokens)/paged.LongAvgLatency.Seconds(),
		((float64(paged.LongAvgTokens)/paged.LongAvgLatency.Seconds())-(float64(off.LongAvgTokens)/off.LongAvgLatency.Seconds()))/(float64(off.LongAvgTokens)/off.LongAvgLatency.Seconds())*100)

	fmt.Println("└─────────────────────┴──────────────────────┴──────────────────────┴──────────────┘")

	fmt.Println("\nThis tests PagedAttention's continuous batching with variable-length requests.")
	fmt.Println("Key advantage: Efficient scheduling of mixed-length KV cache blocks.")
}

func checkServer() bool {
	resp, err := http.Get(baseURL + "/api/tags")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: benchmark_mixed_length <official|paged>")
		fmt.Println("\nTests PagedAttention continuous batching advantage with:")
		fmt.Println("  - Short prompts (~50 tokens)")
		fmt.Println("  - Medium prompts (~200 tokens)")
		fmt.Println("  - Long prompts (~600 tokens)")
		fmt.Println("\nDistribution: 50% short, 33% medium, 17% long")
		os.Exit(1)
	}

	label := os.Args[1]

	if !checkServer() {
		fmt.Printf("\033[0;31mError: Ollama server not running at %s\033[0m\n", baseURL)
		fmt.Println("Start with: ollama serve")
		os.Exit(1)
	}

	fmt.Printf("\n%s=== Mixed Length Benchmark: %s ===%s\n", "\033[0;34m", label, "\033[0m")

	// Test with moderate concurrency
	result := runMixedLengthTest(label, 8, 60)

	// Save results
	data, _ := json.MarshalIndent(result, "", "  ")
	filename := fmt.Sprintf("benchmark_mixed_length_%s_%s.json", label, time.Now().Format("20060102_150405"))
	os.WriteFile(filename, data, 0644)
	fmt.Printf("\nResults saved to: %s\n", filename)
}

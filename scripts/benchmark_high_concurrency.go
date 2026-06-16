// High concurrency benchmark for PagedAttention vs Official Ollama
// Tests memory efficiency and batch processing under high load
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
	testPrompt  = "Write a brief explanation of: "
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

type TestConfig struct {
	Label     string
	Concurrency int
	Requests   int
}

type ConcurrencyResult struct {
	Label              string
	Concurrency        int
	TotalRequests      int
	SuccessCount       int32
	FailureCount       int32
	TotalTime          time.Duration
	AvgLatency         time.Duration
	MinLatency         time.Duration
	MaxLatency         time.Duration
	ThroughputTPS      float64
	AvgTokensPerReq    float64
}

var (
	topics = []string{
		"machine learning",
		"neural networks",
		"deep learning",
		"natural language processing",
		"computer vision",
		"reinforcement learning",
		"transformers",
		"attention mechanisms",
		"gradient descent",
		"backpropagation",
		"convolutional networks",
		"recurrent networks",
		"GANs",
		"autoencoders",
		"transfer learning",
		"fine-tuning",
	}
)

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

func generateRequest(topic string) (*GenerateResponse, time.Duration, error) {
	req := GenerateRequest{
		Model:  model,
		Prompt: testPrompt + topic,
		Stream: false,
	}

	body, _ := json.Marshal(req)
	start := time.Now()

	resp, err := http.Post(baseURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, 0, err
	}

	latency := time.Since(start)
	return &result, latency, nil
}

func warmup(concurrency int) {
	fmt.Printf("  Warming up (%d concurrent)...", concurrency)
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			topic := topics[idx%len(topics)]
			generateRequest(topic)
		}(i)
	}
	wg.Wait()
	fmt.Println(" done")
}

func runConcurrencyTest(config TestConfig) *ConcurrencyResult {
	fmt.Printf("\n%s=== Test: %s (Concurrency: %d, Requests: %d) ===%s\n",
		"\033[0;34m", config.Label, config.Concurrency, config.Requests, "\033[0m")

	// Warmup
	warmup(min(config.Concurrency, 4))

	// Set up rate limiting
	semaphore := make(chan struct{}, config.Concurrency)
	var wg sync.WaitGroup

	var successCount atomic.Int32
	var failureCount atomic.Int32
	var totalLatency atomic.Int64
	var minLatency atomic.Int64
	minLatency.Store(-1)
	var maxLatency atomic.Int64
	var totalTokens atomic.Int32

	start := time.Now()

	for i := 0; i < config.Requests; i++ {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire

		go func(reqID int) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release

			topic := topics[reqID%len(topics)]
			resp, latency, err := generateRequest(topic)

			if err != nil {
				failureCount.Add(1)
				return
			}

			successCount.Add(1)
			totalTokens.Add(int32(countTokens(resp.Response)))

			latencyNs := latency.Nanoseconds()
			totalLatency.Add(latencyNs)

			// Update min
			for {
				current := minLatency.Load()
				if current == -1 || latencyNs < current {
					if minLatency.CompareAndSwap(current, latencyNs) {
						break
					}
				} else {
					break
				}
			}

			// Update max
			for {
				current := maxLatency.Load()
				if latencyNs > current {
					if maxLatency.CompareAndSwap(current, latencyNs) {
						break
					}
				} else {
					break
				}
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(start)

	success := successCount.Load()
	failed := failureCount.Load()

	result := &ConcurrencyResult{
		Label:         config.Label,
		Concurrency:   config.Concurrency,
		TotalRequests: config.Requests,
		SuccessCount:  success,
		FailureCount:  failed,
		TotalTime:     totalTime,
		ThroughputTPS: float64(totalTokens.Load()) / totalTime.Seconds(),
		AvgTokensPerReq: float64(totalTokens.Load()) / float64(success),
	}

	if success > 0 {
		avgNs := totalLatency.Load() / int64(success)
		result.AvgLatency = time.Duration(avgNs)
		result.MinLatency = time.Duration(minLatency.Load())
		result.MaxLatency = time.Duration(maxLatency.Load())
	}

	// Print results
	fmt.Printf("\n  Results:\n")
	fmt.Printf("    Total time: %v\n", totalTime)
	fmt.Printf("    Success: %d, Failed: %d\n", success, failed)
	fmt.Printf("    Avg latency: %v\n", result.AvgLatency)
	fmt.Printf("    Min/Max latency: %v / %v\n", result.MinLatency, result.MaxLatency)
	fmt.Printf("    Throughput: %.2f tok/s\n", result.ThroughputTPS)
	fmt.Printf("    Avg tokens/req: %.1f\n", result.AvgTokensPerReq)

	if failed > 0 {
		failureRate := 100.0 * float64(failed) / float64(config.Requests)
		fmt.Printf("    Failure rate: %.1f%%\n", failureRate)
	}

	return result
}

func runAllTests(label string, results chan *ConcurrencyResult) {
	configs := []TestConfig{
		{Label: label, Concurrency: 4, Requests: 20},
		{Label: label, Concurrency: 8, Requests: 40},
		{Label: label, Concurrency: 16, Requests: 80},
		{Label: label, Concurrency: 32, Requests: 160},
	}

	for _, config := range configs {
		result := runConcurrencyTest(config)
		results <- result
		time.Sleep(2 * time.Second) // Brief pause between tests
	}
}

func printComparison(officialResults, pagedResults []*ConcurrencyResult) {
	fmt.Printf("\n%s=== High Concurrency Comparison ===%s\n", "\033[0;34m", "\033[0m")
	fmt.Println()

	fmt.Println("┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                         Performance by Concurrency Level                                    │")
	fmt.Println("├──────────┬──────────────────────┬──────────────────────┬──────────────────────────────────────┤")
	fmt.Printf("│ %-8s │ %-20s │ %-20s │ %-36s │\n", "Conc.", "Official", "PagedAttn", "Improvement")
	fmt.Println("├──────────┼──────────────────────┼──────────────────────┼──────────────────────────────────────┤")

	for i := 0; i < len(officialResults); i++ {
		off := officialResults[i]
		paged := pagedResults[i]

		offTPS := off.ThroughputTPS
		pagedTPS := paged.ThroughputTPS
		improvement := ((pagedTPS - offTPS) / offTPS) * 100

		offFailRate := 100.0 * float64(off.FailureCount) / float64(off.TotalRequests)
		pagedFailRate := 100.0 * float64(paged.FailureCount) / float64(paged.TotalRequests)

		fmt.Printf("│ %-8d │ %6.1f tok/s (%4.1f%%F) │ %6.1f tok/s (%4.1f%%F) │ %+6.1f%% (%+5.1f%% fail) │\n",
			off.Concurrency, offTPS, offFailRate, pagedTPS, pagedFailRate,
			improvement, pagedFailRate-offFailRate)
	}

	fmt.Println("└──────────┴──────────────────────┴──────────────────────┴──────────────────────────────────────┘")

	fmt.Println("\nLegend:")
	fmt.Println("  tok/s - tokens per second throughput")
	fmt.Println("  %F - failure rate")
	fmt.Println("  Improvement - PagedAttn vs Official throughput difference")
	fmt.Println()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
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
		fmt.Println("Usage: benchmark_high_concurrency <official|paged>")
		fmt.Println("\nThis tests performance under increasing concurrency levels:")
		fmt.Println("  - 4 concurrent (20 requests)")
		fmt.Println("  - 8 concurrent (40 requests)")
		fmt.Println("  - 16 concurrent (80 requests)")
		fmt.Println("  - 32 concurrent (160 requests)")
		os.Exit(1)
	}

	label := os.Args[1]

	if !checkServer() {
		fmt.Printf("\033[0;31mError: Ollama server not running at %s\033[0m\n", baseURL)
		fmt.Println("Start with: ollama serve")
		os.Exit(1)
	}

	fmt.Printf("\n%s=== High Concurrency Benchmark: %s ===%s\n", "\033[0;34m", label, "\033[0m")
	fmt.Println("Testing increasing concurrency to show PagedAttention advantages")

	results := make(chan *ConcurrencyResult, 4)
	runAllTests(label, results)
	close(results)

	// Collect results
	var allResults []*ConcurrencyResult
	for result := range results {
		allResults = append(allResults, result)
	}

	// Save to file
	data, _ := json.MarshalIndent(allResults, "", "  ")
	filename := fmt.Sprintf("benchmark_high_concurrency_%s_%s.json", label, time.Now().Format("20060102_150405"))
	os.WriteFile(filename, data, 0644)
	fmt.Printf("\nResults saved to: %s\n", filename)
}

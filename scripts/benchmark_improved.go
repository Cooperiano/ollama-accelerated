// Improved benchmark with warmup and multiple runs
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"
)

const (
	baseURL     = "http://localhost:11434"
	model       = "mistral:latest"
	testPrompt  = "Explain the concept of machine learning in about 200 words. Cover supervised and unsupervised learning."
	warmupRuns  = 2     // Warmup requests before measuring
	testRuns    = 5     // Number of test runs to average
	concurrency = 4     // Concurrent requests for batch test
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

type BenchmarkStats struct {
	Min   time.Duration
	Max   time.Duration
	Avg   time.Duration
	TPS   float64
	Tokens int
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

func generateRequest() (*GenerateResponse, time.Duration, error) {
	req := GenerateRequest{
		Model:  model,
		Prompt: testPrompt,
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

func warmup() {
	fmt.Print("  Warming up... ")
	for i := 0; i < warmupRuns; i++ {
		generateRequest()
		fmt.Print(".")
	}
	fmt.Println(" done")
}

func runSingleBenchmark() *BenchmarkStats {
	var latencies []time.Duration
	var totalTokens int

	fmt.Printf("  Running %d single-request tests...\n", testRuns)
	for i := 0; i < testRuns; i++ {
		resp, latency, err := generateRequest()
		if err != nil {
			fmt.Printf("    [%d] Error: %v\n", i+1, err)
			continue
		}

		latencies = append(latencies, latency)
		totalTokens += countTokens(resp.Response)

		tps := float64(countTokens(resp.Response)) / latency.Seconds()
		fmt.Printf("    [%d] %v, %d tokens, %.2f tok/s\n", i+1, latency, countTokens(resp.Response), tps)
	}

	if len(latencies) == 0 {
		return nil
	}

	// Calculate stats
	min := latencies[0]
	max := latencies[0]
	sum := time.Duration(0)

	for _, l := range latencies {
		if l < min {
			min = l
		}
		if l > max {
			max = l
		}
		sum += l
	}

	avg := sum / time.Duration(len(latencies))
	avgTokens := totalTokens / len(latencies)
	avgTPS := float64(avgTokens) / avg.Seconds()

	return &BenchmarkStats{
		Min:   min,
		Max:   max,
		Avg:   avg,
		TPS:   avgTPS,
		Tokens: avgTokens,
	}
}

func runConcurrentBenchmark() *BenchmarkStats {
	fmt.Printf("  Running concurrent test (%d parallel requests)...\n", concurrency)

	// Warmup concurrent requests first
	for i := 0; i < 2; i++ {
		var wg sync.WaitGroup
		for j := 0; j < concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				generateRequest()
			}()
		}
		wg.Wait()
	}

	// Actual test
	var wg sync.WaitGroup
	start := time.Now()
	totalTokens := 0

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, _, err := generateRequest()
			if err == nil && resp != nil {
				totalTokens += countTokens(resp.Response)
			}
		}()
	}

	wg.Wait()
	totalTime := time.Since(start)

	avgTPS := float64(totalTokens) / totalTime.Seconds()

	fmt.Printf("    Total time: %v\n", totalTime)
	fmt.Printf("    Total tokens: %d\n", totalTokens)
	fmt.Printf("    Batch TPS: %.2f tok/s\n", avgTPS)

	return &BenchmarkStats{
		Avg:   totalTime,
		TPS:   avgTPS,
		Tokens: totalTokens / concurrency,
	}
}

func runBenchmark(label string) {
	fmt.Printf("\n%s=== %s ===%s\n", "\033[0;34m", label, "\033[0m")

	// Warmup
	warmup()

	// Single request benchmark
	fmt.Println("\n  Single Request Performance:")
	singleStats := runSingleBenchmark()
	if singleStats == nil {
		fmt.Println("    No valid results")
		return
	}

	fmt.Printf("\n  Summary: Min=%v, Max=%v, Avg=%v\n", singleStats.Min, singleStats.Max, singleStats.Avg)
	fmt.Printf("  Average: %d tokens @ %.2f tok/s\n", singleStats.Tokens, singleStats.TPS)

	// Concurrent benchmark
	fmt.Println("\n  Concurrent Request Performance:")
	concurrentStats := runConcurrentBenchmark()

	// Save results
	result := map[string]interface{}{
		"label":         label,
		"timestamp":     time.Now().Format(time.RFC3339),
		"single":        singleStats,
		"concurrent":    concurrentStats,
	}

	data, _ := json.MarshalIndent(result, "", "  ")
	filename := fmt.Sprintf("benchmark_%s_%s.json", label, time.Now().Format("20060102_150405"))
	os.WriteFile(filename, data, 0644)
	fmt.Printf("\n  Results saved to: %s\n", filename)
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
		fmt.Println("Usage: benchmark_improved <official|paged>")
		fmt.Println("\nMake sure ollama serve is running before starting the benchmark")
		os.Exit(1)
	}

	label := os.Args[1]

	if !checkServer() {
		fmt.Printf("\033[0;31mError: Ollama server not running at %s\033[0m\n", baseURL)
		fmt.Println("Start with: ollama serve")
		os.Exit(1)
	}

	runBenchmark(label)
}

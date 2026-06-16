// Performance comparison: Official Ollama vs PagedAttention
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
	testPrompt  = "Write a concise summary of quantum computing in 100 words."
	numRequests = 10
	concurrency = 4
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

type BenchmarkResult struct {
	Label        string
	SingleLatency time.Duration
	SingleTokens  int
	SingleTPS     float64

	ConcurrentTotalTime time.Duration
	ConcurrentAvgTime   time.Duration
	ConcurrentTPS       float64
}

func countTokens(s string) int {
	// Rough estimation: words ≈ tokens for English
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

func runBenchmark(label string) *BenchmarkResult {
	fmt.Printf("\n%s === Running: %s ===%s\n", "\033[0;34m", label, "\033[0m")

	result := &BenchmarkResult{Label: label}

	// Single request test
	fmt.Printf("  Test 1: Single Request...")
	resp, latency, err := generateRequest()
	if err != nil {
		fmt.Printf(" \033[0;31mFAILED: %v\033[0m\n", err)
		return nil
	}

	result.SingleLatency = latency
	result.SingleTokens = countTokens(resp.Response)
	result.SingleTPS = float64(result.SingleTokens) / latency.Seconds()

	fmt.Printf(" \033[0;32m✓\033[0m\n")
	fmt.Printf("    Latency: %v\n", latency)
	fmt.Printf("    Tokens: %d\n", result.SingleTokens)
	fmt.Printf("    TPS: %.2f tok/s\n", result.SingleTPS)

	// Concurrent test
	fmt.Printf("  Test 2: Concurrent Requests (%d parallel)...", concurrency)
	start := time.Now()

	var wg sync.WaitGroup
	results := make(chan *GenerateResponse, numRequests)
	errors := make(chan error, numRequests)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, _, err := generateRequest()
			if err != nil {
				errors <- err
			} else {
				results <- resp
			}
		}()
	}

	wg.Wait()
	close(results)
	close(errors)

	totalTime := time.Since(start)

	// Collect results
	totalTokens := 0
	for resp := range results {
		totalTokens += countTokens(resp.Response)
	}

	result.ConcurrentTotalTime = totalTime
	result.ConcurrentAvgTime = totalTime / time.Duration(numRequests)
	result.ConcurrentTPS = float64(totalTokens) / totalTime.Seconds()

	fmt.Printf(" \033[0;32m✓\033[0m\n")
	fmt.Printf("    Total time: %v\n", totalTime)
	fmt.Printf("    Avg per request: %v\n", result.ConcurrentAvgTime)
	fmt.Printf("    Batch TPS: %.2f tok/s\n", result.ConcurrentTPS)

	// Check for errors
	errCount := len(errors)
	if errCount > 0 {
		fmt.Printf("    \033[0;33mWarnings: %d errors\033[0m\n", errCount)
	}

	return result
}

func printComparison(official, paged *BenchmarkResult) {
	fmt.Printf("\n%s=== Performance Comparison ===%s\n", "\033[0;34m", "\033[0m")
	fmt.Println()
	fmt.Println("┌─────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                          Summary                                       │")
	fmt.Println("├─────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ %-20s │ %-15s │ %-15s │\n", "Metric", "Official", "PagedAttn")
	fmt.Println("├─────────────────────────────────────────────────────────────────────┤")
	fmt.Printf("│ %-20s │ %-15.2f │ %-15.2f │\n", "Single TPS (tok/s)", official.SingleTPS, paged.SingleTPS)
	fmt.Printf("│ %-20s │ %-15.2f │ %-15.2f │\n", "Batch TPS (tok/s)", official.ConcurrentTPS, paged.ConcurrentTPS)
	fmt.Printf("│ %-20s │ %-15v │ %-15v │\n", "Single Latency", official.SingleLatency, paged.SingleLatency)
	fmt.Println("├─────────────────────────────────────────────────────────────────────┤")

	singleImprovement := ((paged.SingleTPS - official.SingleTPS) / official.SingleTPS) * 100
	batchImprovement := ((paged.ConcurrentTPS - official.ConcurrentTPS) / official.ConcurrentTPS) * 100

	fmt.Printf("│ %-20s │ %-15.1f%% │ %-15.1f%% │\n", "Improvement", singleImprovement, batchImprovement)
	fmt.Println("└─────────────────────────────────────────────────────────────────────┘")

	// Analysis
	fmt.Println()
	fmt.Println("\033[0;33mAnalysis:\033[0m")

	if singleImprovement < 1 && singleImprovement > -1 {
		fmt.Println("  ✓ Single-request performance is equivalent (expected)")
	} else if singleImprovement > 0 {
		fmt.Printf("  → Single-request improved by %.1f%%\n", singleImprovement)
	} else {
		fmt.Printf("  → Single-request regressed by %.1f%% (investigate)\n", -singleImprovement)
	}

	if batchImprovement > 5 {
		fmt.Printf("  → Batch processing improved by %.1f%% (PagedAttention advantage)\n", batchImprovement)
	} else if batchImprovement < -5 {
		fmt.Printf("  → Batch processing regressed by %.1f%% (investigate)\n", -batchImprovement)
	} else {
		fmt.Println("  → Batch processing performance similar")
	}

	fmt.Println()
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
		fmt.Println("Usage: benchmark_compare <official|paged>")
		fmt.Println()
		fmt.Println("Run this twice:")
		fmt.Println("  1. Start official ollama, run: benchmark_compare official")
		fmt.Println("  2. Start paged ollama, run: benchmark_compare paged")
		os.Exit(1)
	}

	label := os.Args[1]

	if !checkServer() {
		fmt.Printf("\033[0;31mError: Ollama server not running at %s\033[0m\n", baseURL)
		fmt.Println("Start with: ollama serve &")
		os.Exit(1)
	}

	result := runBenchmark(label)

	// Save to file
	data, _ := json.MarshalIndent(result, "", "  ")
	filename := fmt.Sprintf("benchmark_%s_%s.json", label, time.Now().Format("20060102_150405"))
	os.WriteFile(filename, data, 0644)
	fmt.Printf("\nResults saved to: %s\n", filename)
}

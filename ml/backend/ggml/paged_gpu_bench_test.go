package ggml

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

func initBackend(tb testing.TB) ml.Backend {
	tb.Helper()

	f, err := os.CreateTemp(tb.TempDir(), "*.bin")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, ggml.KV{"general.architecture": "test"}, nil); err != nil {
		tb.Fatal(err)
	}

	b, err := ml.NewBackend(f.Name(), ml.BackendParams{AllocMemory: true})
	if err != nil {
		tb.Skip("no backend available")
	}
	return b
}

func hasGPUBackend(tb testing.TB) {
	initDevices()
	if len(gpus) == 0 {
		tb.Skip("no GPU backend available")
	}
}

func newTestContext(tb testing.TB, backend ml.Backend) ml.Context {
	tb.Helper()

	ctx := backend.NewContext().Input()

	tb.Cleanup(func() {
		ctx.Close()
	})

	return ctx
}

func numElements(shape ...int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
}

func randFloat(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i%100) / 100.0
	}
	return data
}

// BenchmarkPagedAttentionGPU benchmarks PagedAttention on GPU
func BenchmarkPagedAttentionGPU(b *testing.B) {
	hasGPUBackend(b)

	backend := initBackend(b)
	defer backend.Close()

	testCases := []struct {
		name       string
		headDim    int
		numHeads   int
		numKVHeads int
		seqLen     int
		blockSize  int
		batchSize  int
	}{
		{"Small_7B", 128, 32, 8, 512, 16, 1},
		{"Medium_13B", 128, 40, 10, 1024, 16, 1},
		{"Large_70B", 128, 64, 8, 2048, 16, 1},
		{"MultiBatch_7B", 128, 32, 8, 512, 16, 4},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			ctx := newTestContext(b, backend)

			blocksPerSeq := (tc.seqLen + tc.blockSize - 1) / tc.blockSize
			maxBlocksPerSeq := blocksPerSeq + 4

			query := ctx.FromFloats(randFloat(numElements(tc.headDim, 1, tc.numHeads, tc.batchSize)),
				tc.headDim, 1, tc.numHeads, tc.batchSize)

			numBlocks := maxBlocksPerSeq * tc.batchSize
			key := ctx.FromFloats(randFloat(numElements(tc.headDim, tc.numKVHeads, tc.blockSize, numBlocks)),
				tc.headDim, tc.numKVHeads, tc.blockSize, numBlocks)

			value := ctx.FromFloats(randFloat(numElements(tc.headDim, tc.numKVHeads, tc.blockSize, numBlocks)),
				tc.headDim, tc.numKVHeads, tc.blockSize, numBlocks)

			btData := make([]int32, maxBlocksPerSeq*tc.batchSize)
			for seq := 0; seq < tc.batchSize; seq++ {
				for block := 0; block < maxBlocksPerSeq; block++ {
					if block < blocksPerSeq {
						btData[seq*maxBlocksPerSeq+block] = int32(seq*blocksPerSeq + block)
					} else {
						btData[seq*maxBlocksPerSeq+block] = -1
					}
				}
			}
			blockTables := ctx.FromInts(btData, maxBlocksPerSeq, tc.batchSize)

			slData := make([]int32, tc.batchSize)
			for i := range slData {
				slData[i] = int32(tc.seqLen)
			}
			seqLengths := ctx.FromInts(slData, tc.batchSize)

			scale := 1.0 / float64(tc.headDim)

			pa := query.(ml.PagedAttention)

			// Warmup
			out := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, tc.blockSize)
			ctx.Forward(out)
			ctx.Compute(out)

			b.ResetTimer()
			start := time.Now()
			totalOps := int64(0)

			for i := 0; i < b.N; i++ {
				out := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, tc.blockSize)
				ctx.Forward(out)
				ctx.Compute(out)

				flops := int64(2 * tc.headDim * tc.seqLen * tc.seqLen * tc.numHeads * tc.batchSize)
				totalOps += flops
			}

			elapsed := time.Since(start)
			gflops := float64(totalOps) / (elapsed.Seconds() * 1e9)

			b.ReportMetric(gflops, "GFLOP/s")
			b.ReportMetric(float64(elapsed.Milliseconds())/float64(b.N), "ms/op")
		})
	}
}

// BenchmarkPagedAttentionVsFlashAttention compares PagedAttention (decode) with FlashAttention (prefill).
// FlashAttn_prefill reports GFLOP/s for full-prefill attention to establish a performance baseline.
// PagedAttn_decode reports GFLOP/s for single-token decode attention (the key use case for PagedAttention).
func BenchmarkPagedAttentionVsFlashAttention(b *testing.B) {
	hasGPUBackend(b)

	backend := initBackend(b)
	defer backend.Close()

	if testing.Short() {
		b.Skip("Skipping benchmark in short mode")
	}

	headDim := 128
	numHeads := 32
	numKVHeads := 8
	seqLen := 512
	blockSize := 16
	batchSize := 1
	scale := 1.0 / float64(headDim)
	blocksPerSeq := (seqLen + blockSize - 1) / blockSize
	maxBlocksPerSeq := blocksPerSeq + 4

	b.Run("FlashAttn_prefill", func(b *testing.B) {
		ctx := newTestContext(b, backend)

		query := ctx.FromFloats(randFloat(numElements(headDim, seqLen, numHeads, batchSize)),
			headDim, seqLen, numHeads, batchSize)

		key := ctx.FromFloats(randFloat(numElements(headDim, seqLen, numKVHeads, batchSize)),
			headDim, seqLen, numKVHeads, batchSize)

		value := ctx.FromFloats(randFloat(numElements(headDim, seqLen, numKVHeads, batchSize)),
			headDim, seqLen, numKVHeads, batchSize)

		sdpa := query.(ml.ScaledDotProductAttention)

		out := sdpa.ScaledDotProductAttention(ctx, key, value, nil, nil, nil, scale, false)
		ctx.Forward(out)
		ctx.Compute(out)

		flops := int64(2 * headDim * seqLen * seqLen * numHeads * batchSize)

		b.ResetTimer()
		start := time.Now()
		totalOps := int64(0)
		for i := 0; i < b.N; i++ {
			out := sdpa.ScaledDotProductAttention(ctx, key, value, nil, nil, nil, scale, false)
			ctx.Forward(out)
			ctx.Compute(out)
			totalOps += flops
		}
		elapsed := time.Since(start)
		gflops := float64(totalOps) / (elapsed.Seconds() * 1e9)
		b.ReportMetric(gflops, "GFLOP/s")
		b.ReportMetric(float64(elapsed.Milliseconds())/float64(b.N), "ms/op")
	})

	b.Run("PagedAttn_decode", func(b *testing.B) {
		ctx := newTestContext(b, backend)

		query := ctx.FromFloats(randFloat(numElements(headDim, 1, numHeads, batchSize)),
			headDim, 1, numHeads, batchSize)

		numBlocks := maxBlocksPerSeq
		key := ctx.FromFloats(randFloat(numElements(headDim, numKVHeads, blockSize, numBlocks)),
			headDim, numKVHeads, blockSize, numBlocks)

		value := ctx.FromFloats(randFloat(numElements(headDim, numKVHeads, blockSize, numBlocks)),
			headDim, numKVHeads, blockSize, numBlocks)

		btData := make([]int32, maxBlocksPerSeq)
		for i := 0; i < maxBlocksPerSeq; i++ {
			if i < blocksPerSeq {
				btData[i] = int32(i)
			} else {
				btData[i] = -1
			}
		}
		blockTables := ctx.FromInts(btData, maxBlocksPerSeq, 1)
		seqLengths := ctx.FromInts([]int32{int32(seqLen)}, 1)

		pa := query.(ml.PagedAttention)

		out := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, blockSize)
		ctx.Forward(out)
		ctx.Compute(out)

		flops := int64(2 * headDim * seqLen * numHeads)

		b.ResetTimer()
		start := time.Now()
		totalOps := int64(0)
		for i := 0; i < b.N; i++ {
			out := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, blockSize)
			ctx.Forward(out)
			ctx.Compute(out)
			totalOps += flops
		}
		elapsed := time.Since(start)
		gflops := float64(totalOps) / (elapsed.Seconds() * 1e9)
		b.ReportMetric(gflops, "GFLOP/s")
		b.ReportMetric(float64(elapsed.Milliseconds())/float64(b.N), "ms/op")
	})
}

// TestPagedAttentionGPU verifies PagedAttention works on GPU
func TestPagedAttentionGPU(t *testing.T) {
	hasGPUBackend(t)

	backend := initBackend(t)
	defer backend.Close()

	ctx := newTestContext(t, backend)

	headDim := 128
	numHeads := 4
	numKVHeads := 4
	seqLen := 64
	blockSize := 16
	batchSize := 1
	blocksPerSeq := (seqLen + blockSize - 1) / blockSize
	maxBlocksPerSeq := blocksPerSeq + 2

	query := ctx.FromFloats(randFloat(numElements(headDim, 1, numHeads, batchSize)),
		headDim, 1, numHeads, batchSize)

	numBlocks := maxBlocksPerSeq * batchSize
	key := ctx.FromFloats(randFloat(numElements(headDim, numKVHeads, blockSize, numBlocks)),
		headDim, numKVHeads, blockSize, numBlocks)

	value := ctx.FromFloats(randFloat(numElements(headDim, numKVHeads, blockSize, numBlocks)),
		headDim, numKVHeads, blockSize, numBlocks)

	btData := make([]int32, maxBlocksPerSeq*batchSize)
	for i := 0; i < maxBlocksPerSeq; i++ {
		if i < blocksPerSeq {
			btData[i] = int32(i)
		} else {
			btData[i] = -1
		}
	}
	blockTables := ctx.FromInts(btData, maxBlocksPerSeq, batchSize)

	seqLengths := ctx.FromInts([]int32{int32(seqLen)}, batchSize)

	scale := 1.0 / float64(headDim)

	pa := query.(ml.PagedAttention)
	output := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, blockSize)
	if output == nil {
		t.Fatal("PagedAttention returned nil")
	}

	ctx.Forward(output)
	ctx.Compute(output)

	outputFloats := output.Floats()
	if len(outputFloats) == 0 {
		t.Fatal("Output is empty")
	}

	expectedSize := headDim * numHeads * batchSize
	if len(outputFloats) != expectedSize {
		t.Errorf("Expected output size %d, got %d", expectedSize, len(outputFloats))
	}

	hasNonZero := false
	for _, v := range outputFloats {
		if v != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Output contains only zeros")
	}

	t.Logf("PagedAttention GPU test passed! Output size: %d", len(outputFloats))
}

// TestPagedAttentionGPUProfile profiles PagedAttention on GPU
func TestPagedAttentionGPUProfile(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping profile test in short mode")
	}

	hasGPUBackend(t)

	backend := initBackend(t)
	defer backend.Close()

	configs := []struct {
		name       string
		headDim    int
		numHeads   int
		numKVHeads int
		seqLen     int
		blockSize  int
	}{
		{"Mistral-7B", 128, 32, 8, 512, 16},
		{"Llama-13B", 128, 40, 10, 1024, 16},
		{"Llama-70B", 128, 64, 8, 2048, 16},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			ctx := newTestContext(t, backend)

			blocksPerSeq := (cfg.seqLen + cfg.blockSize - 1) / cfg.blockSize
			maxBlocksPerSeq := blocksPerSeq + 4

			query := ctx.FromFloats(randFloat(numElements(cfg.headDim, 1, cfg.numHeads, 1)),
				cfg.headDim, 1, cfg.numHeads, 1)

			numBlocks := maxBlocksPerSeq
			key := ctx.FromFloats(randFloat(numElements(cfg.headDim, cfg.numKVHeads, cfg.blockSize, numBlocks)),
				cfg.headDim, cfg.numKVHeads, cfg.blockSize, numBlocks)

			value := ctx.FromFloats(randFloat(numElements(cfg.headDim, cfg.numKVHeads, cfg.blockSize, numBlocks)),
				cfg.headDim, cfg.numKVHeads, cfg.blockSize, numBlocks)

			btData := make([]int32, maxBlocksPerSeq)
			for i := 0; i < maxBlocksPerSeq; i++ {
				if i < blocksPerSeq {
					btData[i] = int32(i)
				} else {
					btData[i] = -1
				}
			}
			blockTables := ctx.FromInts(btData, maxBlocksPerSeq, 1)

			seqLengths := ctx.FromInts([]int32{int32(cfg.seqLen)}, 1)

			scale := 1.0 / float64(cfg.headDim)

			pa := query.(ml.PagedAttention)

			// Warmup
			out := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, cfg.blockSize)
			ctx.Forward(out)
			ctx.Compute(out)

			// Profile
			iterations := 100
			start := time.Now()

			for i := 0; i < iterations; i++ {
				output := pa.PagedAttention(ctx, key, value, nil, blockTables, seqLengths, scale, cfg.blockSize)
				ctx.Forward(output)
				ctx.Compute(output)
			}

			elapsed := time.Since(start)
			avgMs := float64(elapsed.Milliseconds()) / float64(iterations)

			flops := float64(2 * cfg.headDim * cfg.seqLen * cfg.seqLen * cfg.numHeads)
			gflops := flops / (elapsed.Seconds() / float64(iterations)) / 1e9

			fmt.Printf("%s: seq_len=%d avg_time=%.2fms %.1f GFLOP/s\n",
				cfg.name, cfg.seqLen, avgMs, gflops)
		})
	}
}

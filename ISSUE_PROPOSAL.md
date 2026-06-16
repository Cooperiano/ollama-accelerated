# Discussion: PagedAttention KV Cache for improved memory management

## Problem

Currently, Ollama uses Causal cache which has limitations when handling variable-length sequences:

1. **Memory Fragmentation**: As sequences grow and shrink, free memory becomes fragmented, reducing effective capacity
2. **Inefficient Allocation**: Finding free slots requires O(n) scanning through the cache
3. **Wasted Memory**: Each sequence needs pre-allocated contiguous space, even if most won't use it
4. **Limited Multi-tenancy**: Cannot efficiently share KV cache across multiple concurrent requests

This becomes problematic with:
- Long-running conversations
- Many concurrent users (batch serving scenarios)
- Variable-length prompts (e.g., RAG, document analysis)

## Why This Matters

1. **Better Memory Utilization**: Block-based allocation reduces fragmentation and increases effective cache capacity
2. **Improved Throughput**: More sequences can fit in the same memory, enabling better batching
3. **Cost Efficiency**: Better memory utilization means cheaper serving at scale
4. **Future-proofing**: Foundation for advanced features like priority scheduling and preemption

## Proposed Solution

A block-based KV cache system inspired by vLLM's approach, but adapted for Ollama:

### Core Components

1. **Block-Based Memory Management**
   - Fixed-size blocks (16 tokens per block, configurable)
   - Virtual-to-physical block mapping per sequence
   - Free block pool with LIFO allocation for cache locality

2. **Integration Points**
   - Drop-in replacement for Causal cache (compatible interface)
   - Works with existing CUDA kernels (pagedattn.cu already present)
   - No API changes required

3. **What's NOT Included**
   - Full vLLM-style PagedAttention kernels (complex, CUDA-only)
   - Multi-GPU support
   - Preemption/swap-out mechanisms

### Architecture

```
Current (Causal Cache):
[Seq 1: contiguous 1024 tokens] [Seq 2: contiguous 1024 tokens] [free...]

Proposed (Paged Cache):
[Block Pool]
  ├─ Seq 1: [Block 5] [Block 12] [Block 3] (non-contiguous)
  ├─ Seq 2: [Block 8] [Block 15] (non-contiguous)
  └─ Free: [Block 1] [Block 2] ... (instant allocation)
```

### Performance Characteristics

| Scenario | Current | Proposed |
|----------|---------|----------|
| Block allocation | O(n) scan | O(1) from pool |
| Memory fragmentation | Yes (external) | Minimal (fixed blocks) |
| Max sequences per GB | ~40 | ~50-60 (better packing) |

## How This Would Be Used

No user-visible changes. This is an internal optimization:
- Users can run more concurrent requests
- Longer conversations don't waste memory
- Server can handle variable-length prompts more efficiently

## Testing Approach

1. **Unit Tests**: Block allocation, mapping, cache operations (already written)
2. **Integration Tests**: Real model inference (already tested on Mistral-7B)
3. **Performance Tests**: Compare memory usage and throughput vs Causal cache
4. **Stress Tests**: Many concurrent requests, long sequences

## Questions for Maintainers

Before implementing, I'd like feedback on:

1. **Is this a priority for Ollama?** Would this be accepted if well-implemented?
2. **Scope**: Should this be a drop-in replacement or a separate option?
3. **CUDA vs CPU**: Is GPU-only acceptable, or must CPU work equally well?
4. **Testing requirements**: What benchmarks would you want to see?
5. **Risks**: What concerns do you have about this approach?

## Context

I've implemented a working version (4333 lines) that compiles and passes tests, but my previous PRs were closed:
- #16252: "Ollama already has continuous batching" (fair point)
- #16255: "Not real PagedAttention" (also fair - this is memory management, not the full algorithm)
- #16337: Closed without feedback

I want to ensure I'm working in the right direction before spending more time on this.

---

**References:**
- vLLM PagedAttention paper: https://arxiv.org/abs/2309.06180
- vLLM implementation: https://github.com/vllm-project/vllm

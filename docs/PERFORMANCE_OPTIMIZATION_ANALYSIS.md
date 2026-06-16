# PagedAttention 性能优化分析与建议

## 当前状态分析

### GPU 配置
- **型号**: NVIDIA RTX 3080
- **架构**: Ampere (Compute Capability 8.6)
- **VRAM**: 10GB (当前空闲 ~19GB 显示可能是系统显存)
- **Tensor Cores**: 支持 (FP16/BF16/INT8/INT4)

### 当前 PagedAttention CUDA 实现

#### 核心参数
```cpp
static constexpr int PAGED_ATTN_THREADS = 128;  // 每个块的线程数
template <int HEAD_DIM, int BLOCK_SIZE>  // 模板化头维度和块大小
```

#### 算法特点
- **Online Softmax**: 增量式计算，适合流式处理
- **块级处理**: 逐块处理 KV cache，支持变长序列
- **F32 精度**: 当前只支持单精度浮点

#### 性能瓶颈分析

1. **精度限制**: F32 计算限制吞吐量
2. **内存带宽**: K/V cache 访问可能存在非合并访问
3. **线程利用率**: 128 线程可能不是最优配置
4. **同步开销**: 每块处理需要多次 `__syncthreads()`

---

## 优化机会

### 1. FP16 混合精度 (优先级: ⭐⭐⭐⭐⭐)

**预期收益**: 1.5-2x 吞吐量提升

**实现方式**:
```cpp
// 添加 FP16 版本的 kernel
template <int HEAD_DIM, int BLOCK_SIZE>
static __global__ void paged_attention_kernel_f16(
        const half * __restrict__ Q,           // 改为 half
        const half * __restrict__ K_cache,     // 改为 half
        const half * __restrict__ V_cache,     // 改为 half
        ...);

// 使用 Tensor Cores 进行计算
// 使用 half2 指令进行向量化的点积运算
```

**关键点**:
- Q/K/V 使用 FP16 存储和计算
- Softmax 的 max/sum 使用 FP32 累积保持精度
- 输出转换回 FP32

**参考**: FlashAttention v2 使用类似策略实现 2x 加速

---

### 2. 向量化内存访问 (优先级: ⭐⭐⭐⭐)

**预期收益**: 10-20% 内存带宽提升

**实现方式**:
```cpp
// 使用 float4/half4 进行合并访问
const float4 *k_data_vec = (const float4 *)k_ptr;

// 确保内存对齐
static_assert(HEAD_DIM % 4 == 0, "HEAD_DIM must be aligned");
```

**关键点**:
- 确保所有全局内存访问都是 128-bit 对齐
- 使用 `ldg` 内置函数利用只读缓存
- 合并 K 和 V 的访问模式

---

### 3. 线程块配置优化 (优先级: ⭐⭐⭐)

**预期收益**: 5-15% 提升

**当前**: 128 线程/块
**建议**: 根据头维度和 GPU 架构动态调整

```cpp
// Ampere (RTX 3080) 最优配置:
// - 小头维度 (<64): 256 线程
// - 中等头维度 (64-128): 192 线程
// - 大头维度 (>128): 128 线程

template<int HEAD_DIM>
struct Config {
    static constexpr int THREADS = HEAD_DIM <= 64 ? 256 :
                                   HEAD_DIM <= 128 ? 192 : 128;
};
```

---

### 4. 请求级批处理 (优先级: ⭐⭐⭐⭐)

**预期收益**: 20-40% 高并发场景提升

**当前**: 每个序列独立处理
**建议**: 将多个序列的同一 head 合并处理

```cpp
// 改为 3D grid: (head, seq, virtual_blocks)
// 每个块处理多个序列的片段
// 充分利用 GPU 的并行能力
```

---

### 5. KV Cache 预取 (优先级: ⭐⭐⭐)

**预期收益**: 10-20% 延迟减少

**实现方式**:
```cpp
// 在计算当前块时，预取下一块的 K/V
__builtin_prefetch(k_cache_next_block);
__builtin_prefetch(v_cache_next_block);

// 或使用异步拷贝
cudaMemcpyAsync(&k_cache_next, k_ptr_next, ...);
```

---

### 6. 软件流水线 (优先级: ⭐⭐)

**预期收益**: 5-10% 提升

**实现方式**:
```cpp
// 重叠计算和内存加载
// 1. 当前块计算 attention score
// 2. 同时加载下一块的 K/V
```

---

## 实现优先级排序

### 阶段 1: 快速胜利 (1-2 周)
1. ✅ **FP16 支持** - 最大收益，实现相对简单
2. ✅ **向量化内存访问** - 低风险，稳定收益

### 阶段 2: 架构优化 (2-3 周)
3. **线程块配置优化** - 需要性能测试验证
4. **KV Cache 预取** - 需要仔细设计避免寄存器溢出

### 阶段 3: 高级优化 (4-6 周)
5. **请求级批处理** - 架构级改动
6. **软件流水线** - 复杂度高，需要充分测试

---

## 其他优化维度

### Go 层面优化

1. **调度器优化**
   - 更智能的请求分组
   - 基于序列长度的优先级调度
   - 动态批处理大小调整

2. **内存池管理**
   - 预分配 KV cache 块池
   - 减少运行时分配/释放
   - 块复用策略

3. **CPU-GPU 重叠**
   - 异步调度
   - Pipeline CPU 预处理和 GPU 计算

---

## 性能预估

基于当前 ~90 tok/s 的基线，优化后预期：

| 优化项 | 预期提升 | 实施难度 |
|--------|---------|----------|
| FP16 支持 | +50-100% | 中 |
| 向量化访问 | +10-20% | 低 |
| 线程优化 | +5-15% | 低 |
| 请求批处理 | +20-40% | 高 |
| **总计** | **+85-175%** | - |

**目标**: 达到 150-250 tok/s

---

## 参考实现

- [FlashAttention v2](https://github.com/Dao-AILab/flash-attention) - FP16 混合精度
- [vLLM PagedAttention](https://github.com/vllm-project/vllm) - 生产级实现
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Tensor Core 优化

---

## 下一步行动

1. **性能 Profiling**
   ```bash
   nvidia-smi nvprof --metrics mex::time,...
   ./ollama-paged serve
   ```

2. **FP16 原型开发**
   - 实现 `paged_attention_kernel_f16`
   - 对比 F32 vs F16 性能

3. **基准测试更新**
   - 添加 FP16 性能对比
   - 测试不同量化格式

需要开始实施某个优化吗？

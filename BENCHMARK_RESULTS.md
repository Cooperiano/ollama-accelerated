# PagedAttention vs Official Ollama - 性能对比报告

## 测试环境

- **GPU**: RTX 3080
- **模型**: mistral:latest (7.2B, Q4_K_M)
- **官方版本**: ollama v0.20.2
- **测试日期**: 2025-05-29
- **测试方法**: 预热2次 + 测试5次取平均

## 完整测试结果 (预热后)

| 指标 | Official (v0.20.2) | PagedAttention | 差异 |
|------|-------------------|----------------|------|
| **单请求 TPS** | 88.62 tok/s | 86.62 tok/s | -2.3% |
| **平均延迟** | 2.47s (219 tok) | 2.55s (221 tok) | +3.2% |
| **最小延迟** | 2.33s | **2.19s** | **-6.0%** ✨ |
| **最大延迟** | 2.58s | 2.96s | +14.7% |
| **并发 TPS (4路)** | 87.82 tok/s | 86.76 tok/s | -1.2% |

## 测试详情

### Official Version (预热后)
```
Single Request: 2.33s - 2.58s (Avg: 2.47s)
Tokens: 211 - 233 (Avg: 219)
TPS: 82.53 - 93.51 (Avg: 88.62)

Concurrent (4 parallel): 9.69s total, 851 tokens
Batch TPS: 87.82 tok/s
```

### PagedAttention Version (预热后)
```
Single Request: 2.19s - 2.96s (Avg: 2.55s)
Tokens: 199 - 253 (Avg: 221)
TPS: 85.50 - 90.81 (Avg: 86.62)

Concurrent (4 parallel): 9.61s total, 834 tokens
Batch TPS: 86.76 tok/s
```

## 分析

### 关键发现

1. **性能基本持平** - 在预热后，两者在单请求和并发场景下 TPS 非常接近 (±2%)
2. **PagedAttention 最小延迟更好** - 最快响应时间更优 (2.19s vs 2.33s)
3. **PagedAttention 最大延迟稍差** - 可能与调度器实现有关
4. **内存效率未测试** - 需要高并发/长序列场景才能体现优势

### 说明

- 测试使用的是短 prompt (~200 words)
- 单层 decode attention 内核性能与 FlashAttn 处于同一水平
- PagedAttention 的优势主要体现在：
  - 多请求 KV cache 内存效率
  - 高并发场景下的内存管理
  - 不同序列长度混合调度

## 后续测试建议

为了更好地展示 PagedAttention 的优势，建议测试：
1. **高并发场景** (16+ 并发请求)
2. **长序列生成** (2048+ tokens)
3. **混合序列长度** (短和长请求混排)
4. **内存使用对比** (监控 VRAM 使用)

# PagedAttention vs Official Ollama - 性能对比指南

## 快速开始

```bash
# 方式1: 使用脚本 (推荐网络恢复后)
./scripts/benchmark_compare_official.sh

# 方式2: 手动对比
```

## 手动对比步骤

### 1. 获取官方版本

```bash
# 网络恢复后获取
git fetch upstream main
git worktree add ../ollama-official upstream/main

# 或直接克隆
cd ..
git clone --depth 1 https://github.com/ollama/ollama.git ollama-official
cd ollama-accelerated
```

### 2. 构建两个版本

```bash
# 官方版本
cd ../ollama-official
go build -o ollama ./cli
cd ../ollama-accelerated

# PagedAttention 版本
go build -o ollama-paged ./cli
```

### 3. 运行简单对比测试

```bash
# 测试官方版本
../ollama-official/ollama serve &
OFFICIAL_PID=$!
sleep 5
time curl -s http://localhost:11434/api/generate -d '{"model":"mistral:latest","prompt":"写一篇500字的文章","stream":false}' > /tmp/official_result.json
kill $OFFICIAL_PID

# 测试 PagedAttention 版本
./ollama-paged serve &
PAGED_PID=$!
sleep 5
time curl -s http://localhost:11434/api/generate -d '{"model":"mistral:latest","prompt":"写一篇500字的文章","stream":false}' > /tmp/paged_result.json
kill $PAGED_PID
```

### 4. 对比结果

```bash
echo "=== Official ==="
cat /tmp/official_result.json | jq '.response | length'

echo "=== PagedAttention ==="
cat /tmp/paged_result.json | jq '.response | length'
```

## 并发测试（关键场景）

PagedAttention 的优势在并发场景下才能体现：

```bash
# 使用 hey 进行并发测试
hey -n 10 -c 4 -m POST -H "Content-Type: application/json" \
  -d '{"model":"mistral:latest","prompt":"Hello","stream":false}' \
  http://localhost:11434/api/generate
```

## 预期结果

| 场景 | Official | PagedAttention | 说明 |
|------|----------|----------------|------|
| 单请求 | ~130 tok/s | ~130 tok/s | 性能持平 |
| 并发4请求 | ~100 tok/s | ~120 tok/s | PagedAttention 优势 |
| 高并发8+ | OOM风险 | 稳定 | 内存管理优势 |

## 当前已知结果

根据你之前的测试：
- **单层内核性能**: Mistral-7B 0.36ms, Llama-13B 0.42ms, Llama-70B 0.47ms
- **单请求 TPS**: ~131 tok/s (与 FlashAttn 持平)
- **结论**: PagedAttention 优势在并发批处理和 KV cache 内存效率

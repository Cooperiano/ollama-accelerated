#!/bin/bash
# Benchmark: PagedAttention vs Official Ollama
# Compares single-request and concurrent performance

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Ollama Performance Comparison ===${NC}"
echo "Comparing: Official vs PagedAttention"
echo ""

# Configuration
MODEL=${MODEL:-"mistral:latest"}
PROMPT=${PROMPT:-"Write a 500 word essay about artificial intelligence"}
CONCURRENT_REQUESTS=${CONCURRENT_REQUESTS:-4}
OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR"

# Use system ollama as official baseline if available
SYSTEM_OLLAMA=$(which ollama 2>/dev/null)
if [ -n "$SYSTEM_OLLAMA" ] && [ -x "$SYSTEM_OLLAMA" ]; then
    OFFICIAL_BIN="$SYSTEM_OLLAMA"
    OFFICIAL_VERSION=$($SYSTEM_OLLAMA --version 2>/dev/null | head -1 || echo "unknown")
    echo -e "${GREEN}Using system ollama: $OFFICIAL_BIN ($OFFICIAL_VERSION)${NC}"
else
    OFFICIAL_BIN="./ollama-official/ollama"
fi

PAGED_BIN="./ollama-paged"

echo -e "${YELLOW}1. Checking builds...${NC}"
if [ ! -f "$PAGED_BIN" ]; then
    echo -e "${RED}Error: PagedAttention binary not found at $PAGED_BIN${NC}"
    echo "Run: go build -o ollama ./cli"
    exit 1
fi

# Check for official build
if [ ! -f "$OFFICIAL_BIN" ]; then
    echo -e "${YELLOW}Official build not found. Setting up...${NC}"

    # Check if upstream is available
    if git remote | grep -q upstream; then
        echo "Fetching upstream..."
        git fetch upstream main || echo -e "${RED}Failed to fetch upstream. Network issue?${NC}"

        # Create temporary worktree for official build
        WORKTREE="./.git/worktrees/official"
        if [ ! -d "$WORKTREE" ]; then
            git worktree add ./ollama-official upstream/main -b official-build
        fi

        cd ollama-official
        echo -e "${YELLOW}Building official version...${NC}"
        go build -o ollama ../cli
        cd ..
    else
        echo -e "${YELLOW}No upstream remote. Clone official repo...${NC}"
        [ ! -d ollama-official ] && git clone --depth 1 https://github.com/ollama/ollama.git ollama-official
        cd ollama-official
        go build -o ollama ./cli
        cd ..
    fi
fi

echo -e "${GREEN}Builds ready:${NC}"
echo "  Official: $OFFICIAL_BIN"
echo "  PagedAttn: $PAGED_BIN"
echo ""

# Function to kill background ollama
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    for pid in $(pgrep -f "ollama serve" || true); do
        kill $pid 2>/dev/null || true
    done
    sleep 2
}

trap cleanup EXIT

# Function to run benchmark
run_benchmark() {
    local BIN_PATH=$1
    local LABEL=$2
    local OUTPUT_FILE="$OUTPUT_DIR/benchmark_${LABEL}_${TIMESTAMP}.json"

    echo -e "${BLUE}Running: $LABEL${NC}"

    # Start ollama serve
    cleanup
    echo "Starting server: $BIN_PATH serve"
    $BIN_PATH serve > /dev/null 2>&1 &
    SERVE_PID=$!
    sleep 5  # Wait for server startup

    # Check if server is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        echo -e "${RED}Server failed to start${NC}"
        return 1
    fi

    echo -e "${GREEN}Server ready (PID: $SERVE_PID)${NC}"

    # Test 1: Single request latency
    echo -e "${YELLOW}  Test 1: Single Request${NC}"
    START_TIME=$(date +%s%3N)
    OUTPUT=$(curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"$PROMPT\",
        \"stream\": false
    }")
    END_TIME=$(date +%s%3N)
    SINGLE_TIME=$((END_TIME - START_TIME))

    # Count tokens from response
    TOKEN_COUNT=$(echo "$OUTPUT" | jq -r '.response | length' 2>/dev/null || echo "0")
    WORD_COUNT=$(echo "$OUTPUT" | jq -r '.response | split(" ") | length' 2>/dev/null || echo "0")

    echo "    Time: ${SINGLE_TIME}ms"
    echo "    Tokens: ~$TOKEN_COUNT"
    echo "    TPS: $(echo "scale=2; $TOKEN_COUNT * 1000 / $SINGLE_TIME" | bc) tok/s"

    # Test 2: Concurrent requests (simulate batch)
    echo -e "${YELLOW}  Test 2: Concurrent Requests ($CONCURRENT_REQUESTS parallel)${NC}"
    START_TIME=$(date +%s%3N)

    for i in $(seq 1 $CONCURRENT_REQUESTS); do
        curl -s http://localhost:11434/api/generate -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"Write a short paragraph about topic $i\",
            \"stream\": false
        }" > /dev/null &
    done
    wait

    END_TIME=$(date +%s%3N)
    CONCURRENT_TIME=$((END_TIME - START_TIME))
    CONCURRENT_TPS=$(echo "scale=2; $CONCURRENT_REQUESTS * 50 * 1000 / $CONCURRENT_TIME" | bc)

    echo "    Total time: ${CONCURRENT_TIME}ms"
    echo "    Avg per request: $((CONCURRENT_TIME / CONCURRENT_REQUESTS))ms"
    echo "    Batch TPS: ~${CONCURRENT_TPS} tok/s"

    # Save results
    cat > "$OUTPUT_FILE" <<EOF
{
  "label": "$LABEL",
  "timestamp": "$TIMESTAMP",
  "model": "$MODEL",
  "single_request": {
    "time_ms": $SINGLE_TIME,
    "tokens": $TOKEN_COUNT,
    "tps": $(echo "scale=2; $TOKEN_COUNT * 1000 / $SINGLE_TIME" | bc)
  },
  "concurrent": {
    "requests": $CONCURRENT_REQUESTS,
    "total_time_ms": $CONCURRENT_TIME,
    "avg_per_request_ms": $((CONCURRENT_TIME / CONCURRENT_REQUESTS)),
    "batch_tps": $CONCURRENT_TPS
  }
}
EOF

    kill $SERVE_PID 2>/dev/null || true
    sleep 2

    echo -e "${GREEN}  ✓ $LABEL complete${NC}"
    echo ""
}

# Run benchmarks
run_benchmark "$OFFICIAL_BIN" "official"
run_benchmark "$PAGED_BIN" "paged"

# Compare results
echo -e "${BLUE}=== Results Comparison ===${NC}"

OFFICIAL_RESULT="$OUTPUT_DIR/benchmark_official_${TIMESTAMP}.json"
PAGED_RESULT="$OUTPUT_DIR/benchmark_paged_${TIMESTAMP}.json"

if [ -f "$OFFICIAL_RESULT" ] && [ -f "$PAGED_RESULT" ]; then
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│                    Performance Summary                         │"
    echo "├─────────────────────────────────────────────────────────────┤"

    # Extract metrics
    OFFICIAL_SINGLE_TPS=$(jq -r '.single_request.tps' "$OFFICIAL_RESULT")
    PAGED_SINGLE_TPS=$(jq -r '.single_request.tps' "$PAGED_RESULT")
    OFFICIAL_BATCH_TPS=$(jq -r '.concurrent.batch_tps' "$OFFICIAL_RESULT")
    PAGED_BATCH_TPS=$(jq -r '.concurrent.batch_tps' "$PAGED_RESULT")

    printf "│ %-20s │ %15s │ %15s │\n" "Metric" "Official" "PagedAttn"
    echo "├─────────────────────────────────────────────────────────────┤"
    printf "│ %-20s │ %15s │ %15s │\n" "Single TPS" "${OFFICIAL_SINGLE_TPS} tok/s" "${PAGED_SINGLE_TPS} tok/s"
    printf "│ %-20s │ %15s │ %15s │\n" "Batch TPS" "${OFFICIAL_BATCH_TPS} tok/s" "${PAGED_BATCH_TPS} tok/s"

    # Calculate improvement
    SINGLE_IMPROVEMENT=$(echo "scale=2; ($PAGED_SINGLE_TPS - $OFFICIAL_SINGLE_TPS) / $OFFICIAL_SINGLE_TPS * 100" | bc)
    BATCH_IMPROVEMENT=$(echo "scale=2; ($PAGED_BATCH_TPS - $OFFICIAL_BATCH_TPS) / $OFFICIAL_BATCH_TPS * 100" | bc)

    echo "├─────────────────────────────────────────────────────────────┤"
    printf "│ %-20s │ %15s │ %15s │\n" "Improvement" "${SINGLE_IMPROVEMENT}%" "${BATCH_IMPROVEMENT}%"
    echo "└─────────────────────────────────────────────────────────────┘"

    # Detailed output
    echo ""
    echo "Full results saved to:"
    echo "  - $OFFICIAL_RESULT"
    echo "  - $PAGED_RESULT"
fi

echo -e "${GREEN}=== Benchmark Complete ===${NC}"

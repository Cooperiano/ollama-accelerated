#!/bin/bash
# Quick comparison: System Ollama vs PagedAttention build

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== PagedAttention Performance Comparison ===${NC}\n"

# Check PagedAttention build
if [ ! -f "./ollama-paged" ]; then
    echo -e "${YELLOW}Building PagedAttention version...${NC}"
    go build -o ollama-paged .
fi

# Find system ollama
SYSTEM_OLLAMA=$(which ollama 2>/dev/null)
if [ -z "$SYSTEM_OLLAMA" ]; then
    echo -e "${RED}Error: System ollama not found${NC}"
    echo "Install with: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

SYSTEM_VERSION=$($SYSTEM_OLLAMA --version 2>/dev/null | head -1 || echo "unknown")

echo "System Ollama: $SYSTEM_OLLAMA ($SYSTEM_VERSION)"
echo "Paged Build:   $(pwd)/ollama-paged"
echo ""

# Function to run benchmark
run_benchmark() {
    local BIN=$1
    local LABEL=$2

    echo -e "${YELLOW}Testing: $LABEL${NC}"

    # Kill any running ollama
    pkill -f "ollama serve" || true
    sleep 2

    # Start server
    echo "  Starting server: $BIN"
    $BIN serve > /tmp/ollama_$LABEL.log 2>&1 &
    SERVE_PID=$!

    # Wait for server ready
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}Server failed to start${NC}"
        cat /tmp/ollama_$LABEL.log
        return 1
    fi

    echo -e "${GREEN}  Server ready (PID: $SERVE_PID)${NC}"

    # Run benchmark
    ./scripts/benchmark_compare "$LABEL"

    # Cleanup
    kill $SERVE_PID 2>/dev/null || true
    sleep 2
    echo ""
}

# Run comparison
mkdir -p benchmark_results

echo -e "${BLUE}--- Step 1: Testing Official Version ---${NC}"
run_benchmark "$SYSTEM_OLLAMA" "official"

echo -e "${BLUE}--- Step 2: Testing PagedAttention Version ---${NC}"
run_benchmark "./ollama-paged" "paged"

# Find latest results
OFFICIAL_JSON=$(ls -t benchmark_official_*.json 2>/dev/null | head -1)
PAGED_JSON=$(ls -t benchmark_paged_*.json 2>/dev/null | head -1)

if [ -n "$OFFICIAL_JSON" ] && [ -n "$PAGED_JSON" ]; then
    echo -e "${BLUE}=== Final Comparison ===${NC}"

    # Extract key metrics
    OFFICIAL_TPS=$(jq -r '.singleTPS' "$OFFICIAL_JSON" 2>/dev/null)
    PAGED_TPS=$(jq -r '.singleTPS' "$PAGED_JSON" 2>/dev/null)

    OFFICIAL_BATCH=$(jq -r '.concurrentTPS' "$OFFICIAL_JSON" 2>/dev/null)
    PAGED_BATCH=$(jq -r '.concurrentTPS' "$PAGED_JSON" 2>/dev/null)

    echo ""
    printf "%-20s %-15s %-15s\n" "Metric" "Official" "PagedAttn"
    printf "%-20s %-15s %-15s\n" "--------" "--------" "---------"
    printf "%-20s %-15s %-15s\n" "Single TPS" "${OFFICIAL_TPS} tok/s" "${PAGED_TPS} tok/s"
    printf "%-20s %-15s %-15s\n" "Batch TPS" "${OFFICIAL_BATCH} tok/s" "${PAGED_BATCH} tok/s"
    echo ""

    # Calculate improvement
    if [ "$OFFICIAL_BATCH" != "null" ] && [ "$PAGED_BATCH" != "null" ]; then
        IMPROVEMENT=$(echo "scale=1; ($PAGED_BATCH - $OFFICIAL_BATCH) / $OFFICIAL_BATCH * 100" | bc 2>/dev/null || echo "0")
        echo -e "${GREEN}Batch Improvement: ${IMPROVEMENT}%${NC}"
    fi
fi

echo -e "${GREEN}=== Benchmark Complete ===${NC}"

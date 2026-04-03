#!/bin/bash

# Load environment variables
export $(grep -v '^#' .env | xargs)

run_all_tasks() {
    local api_base=$1
    local model=$2
    local token=$3

    echo ""
    echo "========== EASY (1 stock, 5 days) =========="
    MARKET_TASK=easy MARKET_MAX_STEPS=5 API_BASE_URL=$api_base MODEL_NAME=$model HF_TOKEN=$token python3 inference.py

    echo ""
    echo "========== MEDIUM (3 stocks, 10 days) =========="
    MARKET_TASK=medium MARKET_MAX_STEPS=10 API_BASE_URL=$api_base MODEL_NAME=$model HF_TOKEN=$token python3 inference.py

    echo ""
    echo "========== HARD (5 stocks, 20 days) =========="
    MARKET_TASK=hard MARKET_MAX_STEPS=20 API_BASE_URL=$api_base MODEL_NAME=$model HF_TOKEN=$token python3 inference.py

    echo ""
    echo "========== NIFTY 50 (50 stocks, 30 days) =========="
    MARKET_TASK=nifty50 MARKET_MAX_STEPS=30 API_BASE_URL=$api_base MODEL_NAME=$model HF_TOKEN=$token python3 inference.py
}

echo "================================================"
echo "  Financial Market Environment — Task Runner"
echo "  Server: $ENV_BASE_URL"
echo "================================================"
echo ""
echo "Select model to run:"
echo "  1) Claude Haiku"
echo "  2) Nemotron Ultra (NVIDIA)"
echo "  3) Both"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "## Running: Claude Haiku ##"
        run_all_tasks "https://api.anthropic.com/v1" "claude-haiku-4-5-20251001" "$API_KEY"
        ;;
    2)
        echo ""
        echo "## Running: Nemotron Ultra ##"
        run_all_tasks "https://integrate.api.nvidia.com/v1" "nvidia/llama-3.1-nemotron-ultra-253b-v1" "$NVIDIA_API_KEY"
        ;;
    3)
        echo ""
        echo "## Running: Claude Haiku ##"
        run_all_tasks "https://api.anthropic.com/v1" "claude-haiku-4-5-20251001" "$API_KEY"
        echo ""
        echo "## Running: Nemotron Ultra ##"
        run_all_tasks "https://integrate.api.nvidia.com/v1" "nvidia/llama-3.1-nemotron-ultra-253b-v1" "$NVIDIA_API_KEY"
        ;;
    *)
        echo "Invalid choice. Please enter 1, 2 or 3."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "  Done!"
echo "================================================"

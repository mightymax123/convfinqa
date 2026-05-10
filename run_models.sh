#!/bin/bash
# Run the full evaluation matrix across all supported models and prompting strategies.
# Each model is evaluated on all three strategies with a fixed seed for reproducibility.

set -e

SAMPLE_SIZE=25
SEED=42

MODELS=(
    "openai/gpt-5.4-mini"
    "openai/gpt-5.5"
    "anthropic/claude-haiku-4.5"
    "anthropic/claude-sonnet-4.5"
    "anthropic/claude-sonnet-4.6"
    "google/gemini-3.1-pro-preview"
    "google/gemini-3.1-flash-lite-preview"
)

STRATEGIES=(
    "basic"
    "chain_of_thought"
    "few_shot"
)

for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "Running: model=$model strategy=$strategy"
        docker compose exec app convfinqa \
            --model-name "$model" \
            --prompting-strategy "$strategy" \
            --sample-size "$SAMPLE_SIZE" \
            --seed "$SEED"
    done
done

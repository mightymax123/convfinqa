# ConvFinQA Evaluation Pipeline

[![Python](https://img.shields.io/badge/python-v3.13-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: pyrefly](https://img.shields.io/badge/type%20checked-pyrefly-blue.svg)](https://github.com/facebook/pyrefly)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://github.com/pytest-dev/pytest)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](https://github.com/features/actions)

## Overview

A reproducible evaluation framework for benchmarking Large Language Models on the **ConvFinQA** dataset - a conversational financial question-answering benchmark requiring multi-step reasoning across tabular data, introduced in [Chen et al. (2022)](https://arxiv.org/abs/2210.03849).

This pipeline evaluates models from OpenAI, Anthropic, and Google via **OpenRouter** using three distinct prompting strategies (`basic`, `chain-of-thought`, `few-shot`) with structured output generation and comprehensive accuracy metrics.

### Key Features

- **Multi-Provider Support**: Evaluate models from OpenAI, Anthropic, and Google via OpenRouter
- **Prompt Engineering**: Compare basic, chain-of-thought, and few-shot learning approaches
- **Reproducible Results**: Seeded sampling and containerised environment
- **Production Ready**: Type-safe configuration, exponential-backoff retry logic, structured logging
- **Comprehensive Testing**: Unit tests with pytest, linting with ruff, type checking with pyrefly

## Requirements

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- An [OpenRouter API key](https://openrouter.ai/keys) — used to access models from OpenAI, Anthropic, and Google via a single API

## Setup

1. Initialise the project — copies `sample.env` to `.env` and creates `data/`, `outputs/`, and `logs/` directories:
   ```bash
   make init
   ```
2. Open `.env` and set `OPENROUTER_API_KEY` to your OpenRouter API key.
3. Place the ConvFinQA dataset at `data/convfinqa_dataset.json`.
4. Build and start the container:
   ```bash
   docker compose up --build -d
   ```

## Running Checks

Source `.aliases` once in your shell, then use the shortcuts:

```bash
source .aliases
format       # ruff format + ruff check --fix
code-checks  # ruff format --check + ruff check + pyrefly
run-tests    # coverage run -m pytest + coverage report
pipeline     # code-checks + run-tests (exact mirror of CI)
all-checks   # format + code-checks + run-tests (use this locally before pushing)
```

Or run directly via `docker compose exec`:

```bash
docker compose exec app bash -ic "pipeline"
```

## Running Evaluations

```bash
docker compose exec app convfinqa --model-name <model> --prompting-strategy <strategy> --sample-size <n>
```

Example — `gemini-3.1-flash-lite-preview` with basic prompting, 5 samples:

```bash
docker compose exec app convfinqa --model-name google/gemini-3.1-flash-lite-preview --prompting-strategy basic --sample-size 5
```

| Argument               | Type   | Default                                | Acceptable Values                       | Description                                                        |
| ---------------------- | ------ | -------------------------------------- | --------------------------------------- | ------------------------------------------------------------------ |
| `--model-name`         | string | `google/gemini-3.1-flash-lite-preview` | See supported models table below        | Model to evaluate                                                  |
| `--prompting-strategy` | string | `basic`                                | `basic`, `chain_of_thought`, `few_shot` | Prompting strategy                                                 |
| `--sample-size`        | int    | `10`                                   | any positive integer                    | Number of samples                                                  |
| `--use-train-data`     | bool   | `False`                                | `True`, `False`                         | Use training set instead of test set                               |
| `--seed`               | int    | `None`                                 | any integer                             | Random seed for reproducible sampling (omit for non-deterministic) |

### Supported Models

| Model                                  | Provider  |
| -------------------------------------- | --------- |
| `openai/gpt-4.1`                       | OpenAI    |
| `openai/gpt-4o`                        | OpenAI    |
| `openai/gpt-4o-mini`                   | OpenAI    |
| `openai/o4-mini`                       | OpenAI    |
| `openai/gpt-5.4`                       | OpenAI    |
| `openai/gpt-5.4-mini`                  | OpenAI    |
| `openai/gpt-5.5`                       | OpenAI    |
| `anthropic/claude-sonnet-4.5`          | Anthropic |
| `anthropic/claude-sonnet-4.6`          | Anthropic |
| `google/gemini-3.1-pro-preview`        | Google    |
| `google/gemini-3.1-flash-lite-preview` | Google    |

Results are written to `outputs/<model>_<strategy>/`:
- `convfinqa_responses.json` — per-conversation details
- `eval.txt` — summary with accuracy

## Environment Variables

| Variable             | Default                | Description                                                                                          |
| -------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY` | *(required)*           | Your OpenRouter API key — grants access to all supported providers (OpenAI, Anthropic, Google, etc.) |
| `MAX_RETRIES`        | `10`                   | Retries for both pydantic-ai tool/validation attempts and OpenAI SDK HTTP retries (429 / 5xx)        |
| `UID`                | *(set by `make init`)* | Host user ID — aligns container's non-root user with host                                            |
| `GID`                | *(set by `make init`)* | Host group ID — aligns container's non-root group with host                                          |

## Results

<div align="center">

| Model       | Best Strategy | Accuracy (%) | Sample Size |
| ----------- | ------------- | ------------ | ----------- |
| **o4-mini** | Few-Shot      | **54.15**    | 20          |
| gpt-4.1     | Few-Shot      | 49.13        | 50          |
| gpt-4o      | Few-Shot      | 35.30        | 50          |
| gpt-4o-mini | Few-Shot      | 22.25        | 50          |

</div>

> **Key Insight**: o4-mini significantly outperforms all other models, achieving 54.15% accuracy - over 2x better than gpt-4o-mini and 5% higher than gpt-4.1.

**Performance Comparison**: The chart below shows accuracy across all model-strategy combinations, highlighting Few-Shot learning's consistent superiority and o4-mini's unexpected strong performance.

![ConvFinQA Results - Accuracy by Model and Prompting Strategy](images/accuracy_by_model_strategy.png)

*Full results and analysis available in [REPORT.md](REPORT.md)*

## CI

GitHub Actions runs on every pull request and push to `main`. The workflow builds the dev container and runs the `pipeline` alias inside it:

```
pipeline = code-checks + run-tests
```

Because CI uses the same container and aliases you run locally, there is no drift. If `all-checks` passes locally, CI will pass.

To gate merges on CI, enable branch protection on `main` and require the `checks` status check to pass before merging.

## Architecture & Design

### System Overview

```mermaid
flowchart TD
    A[ConvFinQA Dataset] --> B[Data Parser & Sampler]
    B --> C{Prompting Strategy}
    C -->|Basic| D1[Basic Prompt Builder]
    C -->|Chain-of-Thought| D2[CoT Prompt Builder] 
    C -->|Few-Shot| D3[Few-Shot Prompt Builder]
    D1 --> E[OpenRouter API Client]
    D2 --> E
    D3 --> E
    E -->|Retry Logic| F[LLM Response]
    F --> G[Response Parser]
    G --> H[LLM Judge - Gemini Flash Lite]
    H --> I[Structured Outputs]
    I --> J[JSON Results]
    I --> K[Summary Reports]
    
    style A fill:#e1f5fe
    style E fill:#fff3e0
    style H fill:#f3e5f5
    style I fill:#e8f5e8
```

<details>
<summary>If the diagram above doesn't display properly, click here for a text version</summary>

> **Note**: Some markdown previewers don't support Mermaid diagrams. Here's the same system flow in text format:

```
ConvFinQA Dataset
        ↓
Data Parser & Sampler
        ↓
   Prompting Strategy
    ↙    ↓    ↘
Basic   CoT   Few-Shot
   ↘     ↓     ↙
   OpenAI API Client
    (with retry logic)
        ↓
   LLM Response
         ↓
  Response Parser
         ↓
 LLM Judge (Gemini Flash Lite)
     ↓ concurrent, max 5 at once
 Structured Outputs
    ↙        ↘
JSON Results  Summary Reports
```

</details>

### Prompting Strategies

- **Basic**: Minimal prompt serving as baseline performance
- **Chain-of-Thought**: Step-by-step reasoning before final answers  
- **Few-Shot**: Example-driven learning focusing on output formatting

### Evaluation Methodology

- **Accuracy Metric**: LLM-as-judge (Gemini Flash Lite) evaluates semantic equivalence per answer pair, handling numeric formatting differences and rounding variations
- **Reproducible Sampling**: Configurable sample sizes with optional seeding
- **Structured Outputs**: JSON format with conversation metadata and evaluation results

### Sequential Processing

Conversations are processed one at a time during the generation phase. Each conversation is inherently sequential — the model must wait for each tool call result before deciding the next step. Adding concurrency here would increase rate-limit errors without improving throughput.

The evaluation (judge) phase runs concurrently, capped at 5 simultaneous calls via `asyncio.Semaphore`. This keeps evaluation fast while preventing thundering-herd rate-limit collisions at large scale.

## Future Work

The following enhancements would further improve the codebase:

- **Open Source Models**: Integration with Llama and Mistral via Ollama or HuggingFace
- **Advanced Prompting**: Template experimentation and prompt optimisation

## Contributing

1. Clone the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and ensure tests pass (`all-checks`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

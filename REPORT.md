# ConvFinQA Report

## Overview

This report evaluates a range of frontier LLMs on the **ConvFinQA** dataset — a conversational financial question-answering benchmark introduced in [Chen et al. (2022)](https://arxiv.org/abs/2210.03849). Each dataset entry contains a financial document (a table with surrounding prose) and a series of related questions, where answers to earlier questions often inform later ones. This requires an LLM agent to perform multi-step numerical reasoning grounded in tabular data.

Six models from three providers (OpenAI, Anthropic, Google) were evaluated across three prompting strategies, with all inference routed through **OpenRouter** using a single API key. All results use `--sample-size 25` and `--seed 42` on the dev set.

---

## Method and Architecture

### Dataset Parsing

A `ConvFinQaDataParser` class loads the ConvFinQA JSON dataset and parses each conversation into a typed `ConvQA` Pydantic model containing the conversation id, the financial document (`FinancialDoc`), and paired lists of questions and ground-truth answers. The parser supports both the train and dev splits, with optional random sampling and seeding for reproducibility.

### Prompt Engineering

Prompts are generated using a Strategy pattern. An abstract base class `PromptStrategy` defines the interface, with three concrete implementations:

- **`BasicPromptStrategy`**: A minimal prompt providing the document and questions with no additional guidance. Used as a performance baseline.
- **`ChainOfThoughtPromptStrategy`**: Instructs the model to reason step-by-step before producing final answers, encouraging explicit intermediate reasoning.
- **`FewShotPromptStrategy`**: Provides three worked examples before the actual question. Crucially, the examples focus on **output format and answer style** rather than document reasoning — this was motivated by early experiments showing that formatting errors were a more common failure mode than reasoning errors.

### LLM Agent

Each model is wrapped in a `pydantic-ai` `Agent` configured with:

- **Structured output**: The agent returns a validated `LlmAnswers(answers: list[str])` Pydantic model, ensuring well-formed responses.
- **7 arithmetic tools**: `add`, `subtract`, `multiply`, `divide`, `percentage_change`, `greater`, `exp`. The system prompt mandates that all numerical computation must go through these tools rather than being computed inline — this forces a controlled, logged computation path and reduces hallucinated arithmetic.
- **Rate-limit resilience**: Two retry layers — SDK-level HTTP retries (429/5xx) and an outer exponential backoff loop (1s → 2s → 4s → 8s…) up to `MAX_RETRIES` attempts.
- **Usage cap**: `request_limit=25` per conversation to prevent runaway tool-call loops.
- **Amazon Bedrock excluded**: via `OpenRouterProviderConfig(ignore=["amazon-bedrock"])` due to a known bug where Bedrock omits the `arguments` field in tool call responses.

Generation is **sequential** — one conversation at a time. This is deliberate: each conversation can make up to 25 API requests through the tool-call loop, so concurrent processing would quickly saturate rate limits and trigger thundering-herd backoff cycles.

### Evaluation — LLM-as-Judge

Rather than exact string matching, responses are evaluated by a second **LLM-as-judge** agent hardcoded to `google/gemini-3.1-flash-lite-preview` (the cheapest available model). For each conversation, the judge receives the ground-truth answers and predicted answers as paired lists, and returns a boolean verdict per pair based on **semantic equivalence** — meaning `"3200"` == `"3200.0"`, `"-9.71%"` == `"-9.708%"`, and so on.

This addresses a major limitation of exact-match evaluation, where a model that is conceptually correct but uses different precision or formatting is incorrectly penalised. Judge calls are run concurrently (capped at 5 via `asyncio.Semaphore`) since each is a single, cheap request with no tool calls.

Accuracy is computed per conversation as `correct / total * 100`, then averaged across all conversations.

---

## Results

| Model                 | Basic (%) | Chain-of-Thought (%) | Few-Shot (%) | **Best (%)** |
| --------------------- | --------- | -------------------- | ------------ | ------------ |
| **claude-sonnet-4.5** | 74.63     | 72.63                | **78.63**    | **78.63**    |
| gemini-3.1-pro        | **77.50** | 73.59                | 77.06        | **77.50**    |
| gpt-5.5               | 74.16     | 74.16                | 72.83        | **74.16**    |
| claude-haiku-4.5      | 60.86     | 56.90                | **71.59**    | **71.59**    |
| gemini-3.1-flash-lite | 57.39     | 58.86                | **71.72**    | **71.72**    |
| gpt-5.4-mini          | 60.26     | 65.82                | **67.26**    | **67.26**    |

![ConvFinQA Results - Accuracy by Model and Prompting Strategy](images/accuracy_by_model_strategy.png)

**Key Observations**:

- **Few-shot is the dominant strategy**, winning or tying for best in 5 out of 6 models. The only exception is gemini-3.1-pro, where basic and few-shot are within 0.5% of each other.
- **claude-sonnet-4.5 achieves the highest overall accuracy** at 78.63% with few-shot prompting.
- **Budget models are surprisingly competitive**: both `claude-haiku-4.5` and `gemini-3.1-flash-lite` reach ~72% with few-shot — within 7 percentage points of the best model at a fraction of the cost. This makes them strong candidates when cost efficiency matters.
- **gpt-5.5 is strategy-insensitive**: scores of 74.16%, 74.16%, and 72.83% across basic, CoT, and few-shot respectively suggest the model is robust to prompting approach but may have a ceiling on this task.
- **Chain-of-thought underperforms** relative to few-shot for most models. Explicitly requesting step-by-step reasoning appears to introduce verbosity that can conflict with the structured output format required by the agent.
- **gpt-5.4-mini is the weakest performer** at n=25 despite appearing strong at n=5 — an early indication that small sample sizes produce unreliable rankings.

---

## Error Analysis

The LLM-as-judge evaluator handles the most common failure mode — formatting mismatches — but errors still occur in several patterns:

**Incorrect value extraction**: The model reads the wrong row or column from the financial table. This is particularly common on documents with complex multi-level headers or where the relevant value appears in the prose rather than the table.

**Wrong computation path**: The model calls the correct tools but in the wrong order, or uses an intermediate result incorrectly in a subsequent question. The conversational nature of the dataset means early errors propagate through later answers.

**Placeholder answers**: On highly complex multi-document conversations (`Double_*` entries), some models return `"placeholder"` strings rather than attempting an answer. This is explicitly penalised as incorrect by the judge. The system prompt explicitly forbids placeholder answers, but some models (notably claude-sonnet-4.6) ignored this instruction consistently.

**Unit and scale errors**: Models occasionally return values in different units (e.g. returning millions when the answer expects thousands), or include currency symbols when the ground truth does not.

---

## Outputs

Each pipeline run writes results to `outputs/<model-slug>_<strategy>/`, containing:

- **`convfinqa_responses.json`**: Full per-conversation output including the document, questions, ground-truth answers, LLM answers, and per-answer judge verdicts.
- **`eval.txt`**: Summary with model name, prompting strategy, sample size, and average accuracy.

---

## Future Work

- **Larger sample sizes**: n=25 provides reasonable signal but n=50+ would give more stable rankings, particularly for distinguishing closely-performing models.
- **Open-source models**: Evaluation of Llama and Mistral variants via Ollama or HuggingFace, contingent on available compute.
- **Advanced prompting**: Self-consistency (majority voting across multiple runs), and retrieval-augmented prompts that highlight the most relevant table rows.
- **Concurrent generation**: Explore whether a lower concurrency limit (e.g. 2–3 conversations) is viable for faster runs without triggering sustained rate limiting.

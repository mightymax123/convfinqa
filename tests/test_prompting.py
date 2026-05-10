import pytest

from app.models import ConvQA, FinancialDoc, PromptingStrategy
from app.prompting import PromptGenerator

_CONVERSATION = ConvQA(
    id="conv1",
    doc=FinancialDoc(
        pre_text="Example introductory text.",
        post_text="Example trailing text.",
        table={"2023": {"revenue": 100.0}},
    ),
    questions=["What is revenue?", "What is profit?"],
    answers=["Revenue is money in.", "Profit is money left over."],
)


def _generate(strategy: PromptingStrategy) -> str:
    return PromptGenerator(strategy=strategy).generate_prompt(_CONVERSATION)


def test_basic_prompt_contains_doc_and_questions() -> None:
    """
    Given: The basic prompting strategy
    When: generate_prompt is called
    Then: The prompt contains only the document and questions with no extra framing
    """
    prompt = _generate(PromptingStrategy.BASIC)

    assert "Document:" in prompt
    assert "Questions:" in prompt
    assert "What is revenue?" in prompt
    assert "What is profit?" in prompt
    assert "step-by-step" not in prompt
    assert "Here are three example Q&A pairs" not in prompt


def test_chain_of_thought_prompt_contains_reasoning_instruction() -> None:
    """
    Given: The chain-of-thought prompting strategy
    When: generate_prompt is called
    Then: The prompt includes a step-by-step reasoning nudge plus doc and questions
    """
    prompt = _generate(PromptingStrategy.CHAIN_OF_THOUGHT)

    assert "Think through each question step-by-step" in prompt
    assert "Document:" in prompt
    assert "Questions:" in prompt
    assert "What is revenue?" in prompt


def test_few_shot_prompt_contains_examples_and_questions() -> None:
    """
    Given: The few-shot prompting strategy
    When: generate_prompt is called
    Then: The prompt includes example Q&A pairs and the actual doc and questions
    """
    prompt = _generate(PromptingStrategy.FEW_SHOT)

    assert "Here are three example Q&A pairs" in prompt
    assert "Document:" in prompt
    assert "Questions:" in prompt
    assert "What is revenue?" in prompt


def test_prompting_strategy_rejects_invalid_value() -> None:
    """
    Given: A string that is not a valid PromptingStrategy value
    When: Constructing a PromptingStrategy from it
    Then: It should raise a ValueError
    """
    with pytest.raises(ValueError):
        PromptingStrategy("nonsense")

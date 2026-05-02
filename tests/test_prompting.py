import pytest

from app.data_parser import ConvQA, FinancialDoc
from app.prompting import PromptGenerator, PromptingStrategy


@pytest.mark.parametrize(
    "strategy, expected_substring",
    [
        (PromptingStrategy.BASIC, "Answers (as a Python list of strings):"),
        (PromptingStrategy.CHAIN_OF_THOUGHT, "Step-by-step reasoning"),
        (PromptingStrategy.FEW_SHOT, "Answers:"),
    ],
)
def test_prompt_generator_returns_expected_prompt(strategy: PromptingStrategy, expected_substring: str) -> None:
    """
    Given: A PromptGenerator using a specific strategy
    When: generate_prompt is called with a ConvQA object
    Then: The returned prompt should contain strategy-specific instructions
    """
    conversation: ConvQA = ConvQA(
        id="conv1",
        doc=FinancialDoc(
            pre_text="Example introductory text.",
            post_text="Example trailing text.",
            table={"2023": {"revenue": 100.0}},
        ),
        questions=["What is revenue?", "What is profit?"],
        answers=["Revenue is money in.", "Profit is money left over."],
    )
    generator: PromptGenerator = PromptGenerator(strategy=strategy)
    prompt: str = generator.generate_prompt(conversation)

    assert isinstance(prompt, str)
    assert "Document:" in prompt
    assert "Questions:" in prompt
    assert expected_substring in prompt
    assert "What is revenue?" in prompt
    assert "What is profit?" in prompt


def test_prompting_strategy_rejects_invalid_value() -> None:
    """
    Given: A string that is not a valid PromptingStrategy value
    When: Constructing a PromptingStrategy from it
    Then: It should raise a ValueError
    """
    with pytest.raises(ValueError):
        PromptingStrategy("nonsense")

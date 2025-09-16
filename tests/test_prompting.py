import pytest

from src.data_parser import ConvQA
from src.prompting import PromptGenerator


@pytest.mark.parametrize(
    "strategy, expected_substring",
    [
        ("basic", "Answers (as a Python list of strings):"),
        ("chain_of_thought", "Step-by-step reasoning"),
        ("few_shot", "Answers:"),
    ],
)
def test_prompt_generator_returns_expected_prompt(strategy: str, expected_substring: str) -> None:
    """
    GIVEN a PromptGenerator using a specific strategy
    WHEN generate_prompt is called with a ConvQA object
    THEN the returned prompt should contain strategy-specific instructions
    """
    conversation: ConvQA = ConvQA(
        id="conv1",
        doc="Example financial document text.",
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


def test_prompt_generator_invalid_strategy_raises() -> None:
    """
    GIVEN an invalid strategy name not supported by PromptGenerator
    WHEN initializing the PromptGenerator
    THEN it should raise a ValueError listing available strategies
    """
    with pytest.raises(ValueError) as e:
        PromptGenerator(strategy="nonsense")

    assert "Strategy 'nonsense' is not recognized" in str(e.value)

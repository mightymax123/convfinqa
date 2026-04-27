from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.data_parser import ConvQA, FinancialDoc
from app.evaluator import ConversationsEvaluator

_SAMPLE_DOC = FinancialDoc(
    pre_text="Sample pre text.",
    post_text="Sample post text.",
    table={"2023": {"revenue": 100.0}},
)


@pytest.fixture
def perfect_match_conv() -> list[ConvQA]:
    """
    Given: A conversation where all LLM responses match expected answers
    When: Creating test data for evaluation
    Then: Return conversation with 100% accuracy potential
    """
    return [
        ConvQA(
            id="conv-perfect",
            doc=_SAMPLE_DOC,
            questions=["Q1", "Q2"],
            answers=["10", "20"],
            llm_answers=["10", "20"],
        )
    ]


@pytest.fixture
def partial_match_conv() -> list[ConvQA]:
    """
    Given: A conversation where half of LLM responses match expected answers
    When: Creating test data for evaluation
    Then: Return conversation with 50% accuracy potential
    """
    return [
        ConvQA(
            id="conv-partial",
            doc=_SAMPLE_DOC,
            questions=["Q1", "Q2"],
            answers=["10", "20"],
            llm_answers=["10", "WRONG"],
        )
    ]


@pytest.fixture
def no_match_conv() -> list[ConvQA]:
    """
    Given: A conversation where no LLM responses match expected answers
    When: Creating test data for evaluation
    Then: Return conversation with 0% accuracy potential
    """
    return [
        ConvQA(
            id="conv-wrong",
            doc=_SAMPLE_DOC,
            questions=["Q1", "Q2"],
            answers=["10", "20"],
            llm_answers=["WRONG", "WRONG"],
        )
    ]


@patch("app.evaluator.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_all_conversations_100_percent(
    mock_file: MagicMock, mock_makedirs: MagicMock, perfect_match_conv: list[ConvQA]
) -> None:
    """
    Given: A conversation with perfectly matching LLM responses
    When: evaluate_all_conversations is called
    Then: It should return 100.0 accuracy
    """
    evaluator = ConversationsEvaluator(all_convs=perfect_match_conv)
    result: float = evaluator.evaluate_all_conversations()
    assert result == 100.0


@patch("app.evaluator.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_all_conversations_50_percent(
    mock_file: MagicMock, mock_makedirs: MagicMock, partial_match_conv: list[ConvQA]
) -> None:
    """
    Given: A conversation with one correct and one incorrect answer
    When: evaluate_all_conversations is called
    Then: It should return 50.0 accuracy
    """
    evaluator = ConversationsEvaluator(all_convs=partial_match_conv)
    result: float = evaluator.evaluate_all_conversations()
    assert result == 50.0


@patch("app.evaluator.os.makedirs")
@patch("builtins.open", new_callable=mock_open)
def test_evaluate_all_conversations_0_percent(
    mock_file: MagicMock, mock_makedirs: MagicMock, no_match_conv: list[ConvQA]
) -> None:
    """
    Given: A conversation with all answers wrong
    When: evaluate_all_conversations is called
    Then: It should return 0.0 accuracy
    """
    evaluator = ConversationsEvaluator(all_convs=no_match_conv)
    result: float = evaluator.evaluate_all_conversations()
    assert result == 0.0

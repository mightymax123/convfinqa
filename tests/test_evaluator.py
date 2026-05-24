"""
Tests for ConversationsEvaluator in app/evaluator.py.
"""

from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

from app.evaluator import ConversationsEvaluator
from app.judge import build_judge_agent
from app.models import ConvQA, FinancialDoc
from app.settings import Settings

_SAMPLE_DOC = FinancialDoc(
    pre_text="Sample pre text.",
    post_text="Sample post text.",
    table={"2023": {"revenue": 100.0}},
)


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch: pytest.MonkeyPatch):
    """
    GIVEN tests that construct ConversationsEvaluator,
    WHEN any test in this module runs,
    THEN get_settings() is patched so no real .env file is required.
    """
    from app.settings import get_settings

    get_settings.cache_clear()
    dummy = Settings(openrouter_api_key="test-key")
    with patch("app.judge.get_settings", return_value=dummy):
        yield
    get_settings.cache_clear()


@pytest.fixture
def judge_agent():
    """Return a judge agent backed by TestModel to avoid real API calls."""
    return build_judge_agent()


@pytest.fixture
def perfect_match_conv() -> list[ConvQA]:
    """
    GIVEN a conversation where all LLM responses match expected answers,
    WHEN creating test data for evaluation,
    THEN return a conversation with 100% accuracy potential.
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
    GIVEN a conversation where half of LLM responses match expected answers,
    WHEN creating test data for evaluation,
    THEN return a conversation with 50% accuracy potential.
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
    GIVEN a conversation where no LLM responses match expected answers,
    WHEN creating test data for evaluation,
    THEN return a conversation with 0% accuracy potential.
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


async def test_evaluate_all_conversations_100_percent(
    perfect_match_conv: list[ConvQA],
    judge_agent,
) -> None:
    """
    GIVEN a conversation with perfectly matching LLM responses,
    WHEN evaluate_all_conversations is called with a judge that returns all True,
    THEN it should return 100.0 accuracy.
    """
    test_model = TestModel(custom_output_args={"results": [True, True]})
    with judge_agent.override(model=test_model):
        evaluator = ConversationsEvaluator(
            all_convs=perfect_match_conv,
            judge_agent=judge_agent,
        )
        result = await evaluator.evaluate_all_conversations()

    assert result == 100.0
    assert perfect_match_conv[0].judge_verdicts == [True, True]


async def test_evaluate_all_conversations_50_percent(
    partial_match_conv: list[ConvQA],
    judge_agent,
) -> None:
    """
    GIVEN a conversation with one correct and one incorrect answer,
    WHEN evaluate_all_conversations is called with a judge that returns [True, False],
    THEN it should return 50.0 accuracy.
    """
    test_model = TestModel(custom_output_args={"results": [True, False]})
    with judge_agent.override(model=test_model):
        evaluator = ConversationsEvaluator(
            all_convs=partial_match_conv,
            judge_agent=judge_agent,
        )
        result = await evaluator.evaluate_all_conversations()

    assert result == 50.0
    assert partial_match_conv[0].judge_verdicts == [True, False]


async def test_evaluate_all_conversations_0_percent(
    no_match_conv: list[ConvQA],
    judge_agent,
) -> None:
    """
    GIVEN a conversation with all answers wrong,
    WHEN evaluate_all_conversations is called with a judge that returns all False,
    THEN it should return 0.0 accuracy.
    """
    test_model = TestModel(custom_output_args={"results": [False, False]})
    with judge_agent.override(model=test_model):
        evaluator = ConversationsEvaluator(
            all_convs=no_match_conv,
            judge_agent=judge_agent,
        )
        result = await evaluator.evaluate_all_conversations()

    assert result == 0.0
    assert no_match_conv[0].judge_verdicts == [False, False]

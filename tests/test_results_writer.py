"""
Tests for ResultsWriter in app/results_writer.py.
"""

import json
from unittest.mock import mock_open, patch

import pytest

from app.models import ConvQA, FinancialDoc, ModelName, PromptingStrategy
from app.results_writer import ResultsWriter

_SAMPLE_DOC = FinancialDoc(
    pre_text="Some introductory text.",
    post_text="Some trailing text.",
    table={"2023": {"revenue": 100.0}},
)

_SAMPLE_CONV = ConvQA(
    id="test-1",
    doc=_SAMPLE_DOC,
    questions=["What is revenue?"],
    answers=["100"],
    llm_answers=["100"],
    judge_verdicts=[True],
)


def _make_file_capture():  # type: ignore[no-untyped-def]
    """Return a fake open() that captures writes per file path."""
    written: dict[str, str] = {}

    def fake_open(path: str, mode: str = "r", **kwargs):  # type: ignore[no-untyped-def]
        handle = mock_open()()
        written[path] = ""

        def write(s: str) -> int:
            written[path] += s
            return len(s)

        handle.write = write
        return handle

    return fake_open, written


@pytest.fixture
def writer() -> ResultsWriter:
    """
    GIVEN valid model, strategy, and sample size,
    WHEN creating a ResultsWriter instance with makedirs patched,
    THEN return a configured writer without touching the filesystem.
    """
    with patch("app.results_writer.os.makedirs"):
        return ResultsWriter(
            model_name=ModelName.GPT_5_4_MINI,
            prompting_strategy=PromptingStrategy.CHAIN_OF_THOUGHT,
            sample_size=5,
        )


class TestResultsWriterInit:
    def test_constructs_correct_responses_path(self, writer: ResultsWriter) -> None:
        """
        GIVEN a writer built with GPT_5_4_MINI and chain_of_thought strategy,
        WHEN checking the _responses_path attribute,
        THEN it should point to the expected subfolder and filename.
        """
        assert writer._responses_path == "/code/outputs/gpt-5.4-mini_chain_of_thought/convfinqa_responses.json"

    def test_constructs_correct_eval_path(self, writer: ResultsWriter) -> None:
        """
        GIVEN a writer built with GPT_5_4_MINI and chain_of_thought strategy,
        WHEN checking the _eval_path attribute,
        THEN it should point to the expected subfolder and filename.
        """
        assert writer._eval_path == "/code/outputs/gpt-5.4-mini_chain_of_thought/eval.txt"

    def test_calls_makedirs_on_output_dir(self) -> None:
        """
        GIVEN valid constructor arguments,
        WHEN ResultsWriter is instantiated,
        THEN os.makedirs is called once with the correct directory and exist_ok=True.
        """
        with patch("app.results_writer.os.makedirs") as mock_makedirs:
            ResultsWriter(
                model_name=ModelName.GPT_5_4_MINI,
                prompting_strategy=PromptingStrategy.BASIC,
                sample_size=10,
            )

        mock_makedirs.assert_called_once_with(
            "/code/outputs/gpt-5.4-mini_basic",
            exist_ok=True,
        )


class TestSaveOutputs:
    def test_saves_responses_as_json(self, writer: ResultsWriter) -> None:
        """
        GIVEN a list of conversations with llm_answers populated,
        WHEN save_outputs is called,
        THEN the responses file is written with the correct JSON structure.
        """
        fake_open, written = _make_file_capture()

        with patch("builtins.open", fake_open):
            writer.save_outputs([_SAMPLE_CONV], accuracy=100.0)

        parsed = json.loads(written[writer._responses_path])

        assert len(parsed) == 1
        assert parsed[0]["id"] == "test-1"
        assert parsed[0]["questions"] == ["What is revenue?"]
        assert parsed[0]["answers"] == ["100"]
        assert parsed[0]["llm_answers"] == ["100"]
        assert parsed[0]["judge_verdicts"] == [True]

    def test_saves_evaluation_report(self, writer: ResultsWriter) -> None:
        """
        GIVEN an accuracy value of 75.5,
        WHEN save_outputs is called,
        THEN the eval file contains the model, strategy, accuracy, and sample_size lines.
        """
        fake_open, written = _make_file_capture()

        with patch("builtins.open", fake_open):
            writer.save_outputs([_SAMPLE_CONV], accuracy=75.5)

        eval_content = written[writer._eval_path]
        assert "Model: openai/gpt-5.4-mini" in eval_content
        assert "Prompting Strategy: chain_of_thought" in eval_content
        assert "Average Accuracy: 75.50%" in eval_content
        assert "sample_size: 5" in eval_content

    def test_raises_on_empty_conversation_list(self, writer: ResultsWriter) -> None:
        """
        GIVEN an empty list of conversations,
        WHEN save_outputs is called,
        THEN a ValueError is raised.
        """
        with pytest.raises(ValueError, match="empty"):
            writer.save_outputs([], accuracy=0.0)

    def test_accuracy_formatted_to_two_decimal_places(self, writer: ResultsWriter) -> None:
        """
        GIVEN an accuracy value with many decimal places,
        WHEN save_outputs is called,
        THEN the written accuracy is rounded to exactly two decimal places.
        """
        fake_open, written = _make_file_capture()

        with patch("builtins.open", fake_open):
            writer.save_outputs([_SAMPLE_CONV], accuracy=33.3333)

        assert "Average Accuracy: 33.33%" in written[writer._eval_path]

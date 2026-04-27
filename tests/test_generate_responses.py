"""
Tests for GetAllLlmResponses in app/generate_responses.py.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from app.data_parser import ConvQA, FinancialDoc
from app.generate_responses import GetAllLlmResponses

_SAMPLE_DOC = FinancialDoc(
    pre_text="Some introductory text.",
    post_text="Some trailing text.",
    table={"2023": {"revenue": 100.0}},
)


@pytest.fixture
def dummy_convqa() -> ConvQA:
    """
    GIVEN sample financial conversation data needed for testing,
    WHEN creating a ConvQA instance,
    THEN return a dummy conversation with example questions and answers.
    """
    return ConvQA(
        id="test-1",
        doc=_SAMPLE_DOC,
        questions=["What is revenue?", "What is profit?"],
        answers=["100", "50"],
    )


@pytest.fixture
def generator() -> GetAllLlmResponses:
    """
    GIVEN a mocked data parser and agent,
    WHEN creating a GetAllLlmResponses instance,
    THEN return an instance with a minimal in-memory conversation list.
    """
    with patch("app.generate_responses.ConvFinQaDataParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser.parse_all_conversations.return_value = []
        mock_parser_cls.return_value = mock_parser
        instance = GetAllLlmResponses(sample_size=0, use_seed=False)
    return instance


class TestGetConvResponse:
    def test_get_conv_response_sets_llm_answers(
        self,
        generator: GetAllLlmResponses,
        dummy_convqa: ConvQA,
    ) -> None:
        """
        GIVEN a TestModel configured to return structured answers,
        WHEN _get_conv_response is called on a conversation,
        THEN conv.llm_answers is populated with the model's answers.
        """
        test_model = TestModel(custom_output_args={"answers": ["42", "84"]})

        with generator.agent.override(model=test_model, tools=[]):
            generator._get_conv_response(dummy_convqa)

        assert dummy_convqa.llm_answers == ["42", "84"]

    def test_get_conv_response_uses_prompt_generator(
        self,
        generator: GetAllLlmResponses,
        dummy_convqa: ConvQA,
    ) -> None:
        """
        GIVEN a mocked prompt generator and a TestModel,
        WHEN _get_conv_response is called,
        THEN the prompt generator is called once with the conversation.
        """
        test_model = TestModel(custom_output_args={"answers": ["42"]})

        with patch.object(generator.prompt_gen, "generate_prompt", return_value="Mocked prompt") as mock_prompt:
            with generator.agent.override(model=test_model, tools=[]):
                generator._get_conv_response(dummy_convqa)

        mock_prompt.assert_called_once_with(dummy_convqa)


class TestGetAllResponses:
    def test_get_all_responses_processes_all_conversations(
        self,
        generator: GetAllLlmResponses,
    ) -> None:
        """
        GIVEN a generator with two conversations and a TestModel,
        WHEN get_all_responses is called,
        THEN all conversations have llm_answers populated.
        """
        conv1 = ConvQA(id="c1", doc=_SAMPLE_DOC, questions=["Q1"], answers=["A1"])
        conv2 = ConvQA(id="c2", doc=_SAMPLE_DOC, questions=["Q2"], answers=["A2"])
        generator.all_convs = [conv1, conv2]

        test_model = TestModel(custom_output_args={"answers": ["answer"]})

        with patch.object(generator, "_save_conversations_to_json"):
            with generator.agent.override(model=test_model, tools=[]):
                result = generator.get_all_responses()

        assert result[0].llm_answers == ["answer"]
        assert result[1].llm_answers == ["answer"]

    def test_get_all_responses_raises_on_failure(
        self,
        generator: GetAllLlmResponses,
    ) -> None:
        """
        GIVEN a conversation that causes the agent to raise an exception,
        WHEN get_all_responses is called,
        THEN a RuntimeError is raised with the conversation ID.
        """
        conv = ConvQA(id="fail-1", doc=_SAMPLE_DOC, questions=["Q?"], answers=["A"])
        generator.all_convs = [conv]

        with patch.object(generator, "_get_conv_response", side_effect=Exception("boom")):
            with pytest.raises(RuntimeError, match="fail-1"):
                generator.get_all_responses()

from unittest.mock import MagicMock, patch

import pytest

from src.data_parser import ConvQA
from src.generate_responses import GetAllLlmResponses


@pytest.fixture
def dummy_convqa() -> ConvQA:
    """
    Provides a dummy ConvQA instance for use in unit tests.

    Returns:
        ConvQA: A sample conversation with example questions and answers.
    """
    return ConvQA(
        id="test-1",
        doc="Some financial report.",
        questions=["What is revenue?", "What is profit?"],
        answers=["100", "50"],
    )


def test_extract_list_from_valid_llm_response() -> (
    None
):  # Testing internal method because it's core to formatting logic
    """
    GIVEN a well-formatted string LLM response
    WHEN _extract_list_from_llm_response is called
    THEN it should return the parsed Python list of strings
    """
    response_handler = GetAllLlmResponses()
    response = "Here are your answers: ['100', '50']"
    result = response_handler._extract_list_from_llm_response(response)

    assert result == ["100", "50"]


def test_extract_list_from_empty_response() -> (
    None
):  # Testing internal method because it's core to formatting logic
    """
    GIVEN an empty string
    WHEN _extract_list_from_llm_response is called
    THEN it should return an empty list
    """
    result = GetAllLlmResponses()._extract_list_from_llm_response("")
    assert result == []


def test_extract_list_from_invalid_list() -> (
    None
):  # Testing internal method because it's core to formatting logic
    """
    GIVEN a badly formatted response that resembles a list but isn't valid Python
    WHEN _extract_list_from_llm_response is called
    THEN it should return an empty list
    """
    invalid_response = (
        "Answers: ['100', 50"  # missing closing bracket, invalid response
    )
    result = GetAllLlmResponses()._extract_list_from_llm_response(invalid_response)
    assert result == []


@patch("src.generate_responses.OpenAiLlmResponse.get_response")
@patch("src.generate_responses.PromptGenerator.generate_prompt")
def test_get_conv_response_calls_llm_and_sets_attributes(
    mock_generate_prompt: MagicMock,
    mock_get_response: MagicMock,
    dummy_convqa: ConvQA,
) -> None:
    """
    GIVEN mocked prompt generator and LLM response
    WHEN get_conv_response is called on a conversation
    THEN it should update the conversation with LLM output and formatted response
    """
    mock_generate_prompt.return_value = "Mocked prompt"
    mock_get_response.return_value = "['42', '84']"

    generator = GetAllLlmResponses()
    generator._get_conv_response(dummy_convqa)

    assert dummy_convqa.llm_response == "['42', '84']"
    assert dummy_convqa.formatted_llm_response == ["42", "84"]
    mock_generate_prompt.assert_called_once_with(dummy_convqa)
    mock_get_response.assert_called_once_with(prompt="Mocked prompt")

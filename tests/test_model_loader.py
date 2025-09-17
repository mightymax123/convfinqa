from unittest.mock import MagicMock, patch

import pytest

from src.model_loader import ModelName, OpenAiLlmResponse


def test_valid_model_initialization() -> None:
    """
    Given: A valid model name supported by the system
    When: An OpenAiLlmResponse is initialized with that model
    Then: It should correctly set the model name to the expected enum value
    """
    llm = OpenAiLlmResponse(model_name="gpt-4o")
    assert llm.model_name == ModelName.GPT_4O.value


def test_invalid_model_raises_error() -> None:
    """
    Given: An invalid or unsupported model name
    When: An OpenAiLlmResponse is initialized with that name
    Then: It should raise a ValueError indicating an invalid model
    """
    with pytest.raises(ValueError) as e:
        OpenAiLlmResponse(model_name="not-a-real-model")
    assert "Invalid model name" in str(e.value)


@patch("src.model_loader.OpenAiLlmResponse.get_response")
def test_get_response_returns_list_of_answers(mock_get_response: MagicMock) -> None:
    """
    Given: The get_response method is mocked to simulate an LLM output
    When: It is called with a structured multi-question prompt
    Then: It should return a mocked list of answers as a string
    """
    mock_response = "['Revenue is money in.', 'Profit is money left over.']"
    mock_get_response.return_value = mock_response

    llm = OpenAiLlmResponse(model_name="gpt-4o")
    prompt = "What is revenue? {next_question} What is profit?"
    result = llm.get_response(prompt)

    assert result == mock_response
    mock_get_response.assert_called_once_with(prompt)

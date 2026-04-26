"""
Tests for OpenAiLlmResponse model loader.
"""

from unittest.mock import MagicMock, patch

from app.model_loader import ModelName, OpenAiLlmResponse


def test_valid_model_initialisation() -> None:
    """
    Given: A valid ModelName enum value
    When: An OpenAiLlmResponse is initialised with that model
    Then: It should correctly set the model name to the enum's string value
    """
    llm = OpenAiLlmResponse(model_name=ModelName.GPT_4O)
    assert llm.model_name == ModelName.GPT_4O.value


@patch("app.model_loader.OpenAiLlmResponse.get_response")
def test_get_response_returns_list_of_answers(mock_get_response: MagicMock) -> None:
    """
    Given: The get_response method is mocked to simulate an LLM output
    When: It is called with a structured multi-question prompt
    Then: It should return a mocked list of answers as a string
    """
    mock_response = "['Revenue is money in.', 'Profit is money left over.']"
    mock_get_response.return_value = mock_response

    llm = OpenAiLlmResponse(model_name=ModelName.GPT_4O)
    prompt = "What is revenue? {next_question} What is profit?"
    result = llm.get_response(prompt)

    assert result == mock_response
    mock_get_response.assert_called_once_with(prompt)

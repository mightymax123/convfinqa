"""
Tests for the pydantic-ai agent setup in app/agent.py.
"""

# Prevent accidental real model requests in this test module.
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel

from app.agent import LlmAnswers, ModelName, build_agent, get_response

_EXPECTED_TOOLS = {"add", "subtract", "multiply", "divide", "percentage_change", "greater", "exp"}


class TestLlmAnswers:
    def test_valid_answers_accepted(self) -> None:
        """
        GIVEN a list of string answers,
        WHEN LlmAnswers is initialised,
        THEN the answers field is set correctly.
        """
        result = LlmAnswers(answers=["10", "20"])

        assert result.answers == ["10", "20"]

    def test_empty_answers_accepted(self) -> None:
        """
        GIVEN an empty list,
        WHEN LlmAnswers is initialised,
        THEN the answers field is an empty list.
        """
        result = LlmAnswers(answers=[])

        assert result.answers == []


class TestBuildAgent:
    def test_build_agent_returns_agent_instance(self) -> None:
        """
        GIVEN a valid model name and max_retries value,
        WHEN build_agent is called,
        THEN an Agent instance is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=2)

        assert isinstance(agent, Agent)

    def test_build_agent_registers_all_tools(self) -> None:
        """
        GIVEN a valid model name and max_retries value,
        WHEN build_agent is called,
        THEN all expected arithmetic tools are registered on the agent.
        """
        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=1)
        registered = set(agent._function_toolset.tools.keys())

        assert registered == _EXPECTED_TOOLS

    def test_build_agent_configures_model(self) -> None:
        """
        GIVEN a valid model name,
        WHEN build_agent is called,
        THEN the agent's model is an OpenAIModel with the correct model name.
        """
        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=5)
        assert isinstance(agent.model, OpenAIModel)
        assert agent.model.model_name == ModelName.GPT_4O.value


class TestGetResponse:
    async def test_get_response_returns_llm_answers(self) -> None:
        """
        GIVEN a TestModel configured to return structured answers,
        WHEN get_response is called with a prompt,
        THEN an LlmAnswers instance with the expected answers is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["42", "84"]})

        with agent.override(model=test_model, tools=[]):
            result = await get_response(agent, "What is revenue? {next_question} What is profit?")

        assert isinstance(result, LlmAnswers)
        assert result.answers == ["42", "84"]

    async def test_get_response_handles_single_answer(self) -> None:
        """
        GIVEN a TestModel configured to return a single answer,
        WHEN get_response is called,
        THEN an LlmAnswers instance with one answer is returned.
        """
        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["100"]})

        with agent.override(model=test_model, tools=[]):
            result = await get_response(agent, "What is the total revenue?")

        assert result is not None
        assert result.answers == ["100"]

    async def test_get_response_returns_none_on_usage_limit_exceeded(self) -> None:
        """
        GIVEN an agent run with a request limit of zero,
        WHEN get_response is called,
        THEN None is returned rather than raising UsageLimitExceeded.
        """
        from unittest.mock import patch

        from pydantic_ai.usage import UsageLimits

        agent = build_agent(model_name=ModelName.GPT_4O, max_retries=1)
        test_model = TestModel(custom_output_args={"answers": ["42"]})

        with agent.override(model=test_model, tools=[]):
            with patch("app.agent.UsageLimits", return_value=UsageLimits(request_limit=0)):
                result = await get_response(agent, "What is revenue?")

        assert result is None

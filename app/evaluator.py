"""
ConversationsEvaluator class for evaluating conversations using an LLM judge.
"""

import asyncio
import os

from loguru import logger
from pydantic_ai import Agent

from app.agent import ModelName
from app.data_parser import ConvQA
from app.judge import JudgeResult, get_judge_response
from app.prompting import PromptingStrategy
from app.settings import get_settings

_MAX_CONCURRENT_JUDGE_CALLS = 5


class ConversationsEvaluator:
    """Evaluates LLM responses against ground-truth answers using an LLM judge."""

    def __init__(
        self,
        all_convs: list[ConvQA],
        model_name: ModelName,
        prompting_strategy: PromptingStrategy,
        sample_size: int,
        judge_agent: Agent[None, JudgeResult],
    ) -> None:
        """Initialise the evaluator with conversations, run configuration, and a judge agent.

        Args:
            all_convs: Conversations whose LLM responses will be evaluated.
            model_name: The model used to generate responses.
            prompting_strategy: The prompting strategy used.
            sample_size: Number of samples that were evaluated.
            judge_agent: Pre-built judge agent used to score answer pairs.
        """
        logger.info(
            f"Initialising ConversationsEvaluator with model: {model_name.value}, "
            f"strategy: {prompting_strategy.value}, sample size: {sample_size}"
        )
        self.all_convs = all_convs
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.sample_size = sample_size
        self.judge_agent = judge_agent
        self.max_retries = get_settings().max_retries
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

        subfolder = f"{model_name.value.split('/')[-1]}_{prompting_strategy.value}"
        self.save_path = os.path.join("/code/outputs", subfolder, "eval.txt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    async def _evaluate_conversation(self, conv: ConvQA) -> float:
        """Compute the accuracy of the LLM responses for a single conversation.

        Calls the judge agent with the ground-truth and predicted answer lists.

        Args:
            conv: The conversation to evaluate.

        Returns:
            Percentage of answers judged correct by the LLM judge.
        """
        total = len(conv.answers)
        if not conv.llm_answers or total == 0:
            return 0.0

        async with self._semaphore:
            judge_result = await get_judge_response(
                self.judge_agent,
                ground_truth=conv.answers,
                predicted=conv.llm_answers,
                max_retries=self.max_retries,
            )

        correct = sum(judge_result.results)
        accuracy = (correct / total) * 100

        logger.debug(f"Evaluated conversation {conv.id}: accuracy = {accuracy:.2f}%")

        return accuracy

    def _save_evaluation(self, accuracy: float) -> None:
        """Write evaluation results to the output file.

        Args:
            accuracy: Average accuracy across all evaluated conversations.
        """
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.model_name.value}\n")
            f.write(f"Prompting Strategy: {self.prompting_strategy.value}\n")
            f.write(f"Average Accuracy: {accuracy:.2f}%\n")
            f.write(f"sample_size: {self.sample_size}\n")

        logger.info(f"Saved evaluation results to {self.save_path}")

    async def evaluate_all_conversations(self) -> float:
        """Evaluate all conversations concurrently and return the average accuracy.

        Returns:
            Average accuracy across all conversations as a percentage.
        """
        accs = await asyncio.gather(*[self._evaluate_conversation(conv) for conv in self.all_convs])
        avg_accuracy = sum(accs) / len(accs) if accs else 0.0

        self._save_evaluation(avg_accuracy)

        logger.info(f"Evaluated {len(self.all_convs)} conversations. Average accuracy: {avg_accuracy:.2f}%")

        return avg_accuracy

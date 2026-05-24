"""
ConversationsEvaluator class for evaluating conversations using an LLM judge.
"""

import asyncio

from loguru import logger
from pydantic_ai import Agent

from app.judge import JudgeResult, get_judge_response
from app.models import ConvQA

_MAX_CONCURRENT_JUDGE_CALLS = 5


class ConversationsEvaluator:
    """Evaluates LLM responses against ground-truth answers using an LLM judge."""

    def __init__(
        self,
        all_convs: list[ConvQA],
        judge_agent: Agent[None, JudgeResult],
    ) -> None:
        """Initialise the evaluator with conversations and a judge agent.

        Args:
            all_convs: Conversations whose LLM responses will be evaluated.
            judge_agent: Pre-built judge agent used to score answer pairs.
        """
        logger.info(f"Initialising ConversationsEvaluator with {len(all_convs)} conversations")
        self.all_convs = all_convs
        self.judge_agent = judge_agent
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_JUDGE_CALLS)

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
            )

        correct = sum(judge_result.results)
        accuracy = (correct / total) * 100

        conv.judge_verdicts = judge_result.results
        logger.debug(f"Evaluated conversation {conv.id}: accuracy = {accuracy:.2f}%")

        return accuracy

    async def evaluate_all_conversations(self) -> float:
        """Evaluate all conversations concurrently and return the average accuracy.

        Returns:
            Average accuracy across all conversations as a percentage.
        """
        accs: list[float] = await asyncio.gather(*[self._evaluate_conversation(conv) for conv in self.all_convs])
        avg_accuracy = sum(accs) / len(accs) if accs else 0.0

        logger.info(f"Evaluated {len(self.all_convs)} conversations. Average accuracy: {avg_accuracy:.2f}%")

        return avg_accuracy

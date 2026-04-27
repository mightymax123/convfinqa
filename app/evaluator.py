"""
ConversationsEvaluator class for evaluating conversations with an LLM.
"""

import os

from loguru import logger

from app.agent import ModelName
from app.data_parser import ConvQA
from app.prompting import PromptingStrategy


class ConversationsEvaluator:
    """Evaluates LLM responses against ground-truth answers for a set of conversations."""

    def __init__(
        self,
        all_convs: list[ConvQA],
        model_name: ModelName = ModelName.GPT_4_1,
        prompting_strategy: PromptingStrategy = PromptingStrategy.CHAIN_OF_THOUGHT,
        sample_size: int = 100,
    ) -> None:
        """Initialise the evaluator with conversations and run configuration.

        Args:
            all_convs: Conversations whose LLM responses will be evaluated.
            model_name: The model used to generate responses.
            prompting_strategy: The prompting strategy used.
            sample_size: Number of samples that were evaluated.
        """
        logger.info(
            f"Initialising ConversationsEvaluator with model: {model_name.value}, "
            f"strategy: {prompting_strategy.value}, sample size: {sample_size}"
        )
        self.all_convs = all_convs
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.sample_size = sample_size

        subfolder = f"{model_name.value}_{prompting_strategy.value}"
        self.save_path = os.path.join("/code/outputs", subfolder, "eval.txt")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _evaluate_conversation(self, conv: ConvQA) -> float:
        """Compute the accuracy of the LLM responses for a single conversation.

        Args:
            conv: The conversation to evaluate.

        Returns:
            Percentage of answers that exactly match the ground truth.
        """
        preds = [pred.strip() for pred in conv.llm_answers if pred is not None]
        true = [ans.strip() for ans in conv.answers if ans is not None]

        total = len(true)

        if not preds or total == 0:
            return 0.0

        correct = sum(1 for t, p in zip(true, preds, strict=False) if t == p)
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

    def evaluate_all_conversations(self) -> float:
        """Evaluate all conversations and return the average accuracy.

        Returns:
            Average accuracy across all conversations as a percentage.
        """
        accs = [self._evaluate_conversation(conv) for conv in self.all_convs]
        avg_accuracy = sum(accs) / len(accs) if accs else 0.0

        self._save_evaluation(avg_accuracy)

        logger.info(f"Evaluated {len(self.all_convs)} conversations. Average accuracy: {avg_accuracy:.2f}%")

        return avg_accuracy

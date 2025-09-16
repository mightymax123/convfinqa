"""
ConversationsEvaluator class for evaluating conversations with an LLM.
"""

import os

from src.data_parser import ConvQA
from src.logger import get_logger

logger = get_logger(__name__)


class ConversationsEvaluator:
    def __init__(
        self,
        all_convs: list[ConvQA],
        model_name: str = "gpt-4.1",
        prompting_strategy: str = "chain_of_thought",
        sample_size: int = 100,
    ):
        """
        Initializes the ConversationsEvaluator with a list of conversations.

        args:
            all_convs (list[ConvQA]): A list of ConvQA objects representing conversations.
        """
        logger.info(
            f"Initializing ConversationsEvaluator with model: {model_name}, strategy: {prompting_strategy}, sample size: {sample_size}"
        )
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.all_convs = all_convs
        self.sample_size = sample_size

        subfolder = f"{model_name}_{prompting_strategy}"

        self.save_path = os.path.join("/app/outputs", subfolder, "eval.txt")

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _evaluate_conversation(self, conv: ConvQA) -> float:
        """
        Evaluates a single conversation and returns the accuracy of the LLM's responses.

        args:
            conv (ConvQA): A ConvQA object representing a conversation.

        returns:
            float: The accuracy of the LLM's responses in the conversation.
        """
        preds = [
            pred.strip() for pred in conv.formatted_llm_response if pred is not None
        ]
        true = [ans.strip() for ans in conv.answers if ans is not None]

        total = len(true)
        correct = 0

        if not preds or total == 0:
            return 0.0

        for t, p in zip(true, preds):
            if t == p:
                correct += 1

        accuracy = (correct / total) * 100

        logger.debug(f"Evaluated conversation {conv.id}: accuracy = {accuracy:.2f}%")

        return accuracy

    def evaluate_all_conversations(self) -> float:
        """
        Evaluates all conversations and returns the average accuracy of the LLM's responses.

        returns:
            float: The average accuracy of the LLM's responses across all conversations.
        """
        accs = [self._evaluate_conversation(conv) for conv in self.all_convs]

        avg_accuracy = sum(accs) / len(accs) if accs else 0.0

        self._save_evaluation(avg_accuracy)

        logger.info(
            f"Evaluated {len(self.all_convs)} conversations. Average accuracy: {avg_accuracy:.2f}%"
        )

        return avg_accuracy

    def _save_evaluation(self, accuracy: float) -> None:
        """
        Saves the evaluation results to a file.

        args:
            accuracy (float): The average accuracy of the LLM's responses.
        """
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Prompting Strategy: {self.prompting_strategy}\n")
            f.write(f"Average Accuracy: {accuracy:.2f}%\n")
            f.write(f"sample_size: {self.sample_size}\n")

        logger.info(f"Saved evaluation results to {self.save_path}")

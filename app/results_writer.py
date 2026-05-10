"""
ResultsWriter: centralised output persistence for the ConvFinQA pipeline.
"""

import json
import os

from loguru import logger

from app.models import ConvQA, ModelName, PromptingStrategy

_OUTPUT_ROOT = "/code/outputs"


class ResultsWriter:
    """Writes pipeline outputs (responses JSON and evaluation report) to disk."""

    def __init__(
        self,
        model_name: ModelName,
        prompting_strategy: PromptingStrategy,
        sample_size: int,
    ) -> None:
        """Construct output paths and create the output directory.

        Args:
            model_name: The model used to generate responses.
            prompting_strategy: The prompting strategy used.
            sample_size: Number of conversations that were evaluated.
        """
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.sample_size = sample_size

        subfolder = f"{model_name.value.split('/')[-1]}_{prompting_strategy.value}"
        output_dir = os.path.join(_OUTPUT_ROOT, subfolder)

        self._responses_path = os.path.join(output_dir, "convfinqa_responses.json")
        self._eval_path = os.path.join(output_dir, "eval.txt")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"ResultsWriter initialised — output directory: {output_dir}")

    def _save_responses(self, all_convs: list[ConvQA]) -> None:
        """Serialise conversations with LLM answers to a JSON file.

        Args:
            all_convs: Conversations with llm_answers populated.

        Raises:
            ValueError: If all_convs is empty.
        """
        if not all_convs:
            raise ValueError("The list of conversations is empty.")

        data = [conv.model_dump() for conv in all_convs]

        logger.info(f"Saving {len(data)} conversations to {self._responses_path}")

        with open(self._responses_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversations saved successfully to {self._responses_path}")

    def _save_evaluation(self, accuracy: float) -> None:
        """Write evaluation summary to the eval report file.

        Args:
            accuracy: Average accuracy across all evaluated conversations.
        """
        with open(self._eval_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.model_name.value}\n")
            f.write(f"Prompting Strategy: {self.prompting_strategy.value}\n")
            f.write(f"Average Accuracy: {accuracy:.2f}%\n")
            f.write(f"sample_size: {self.sample_size}\n")

        logger.info(f"Saved evaluation results to {self._eval_path}")

    def save_outputs(self, all_convs: list[ConvQA], accuracy: float) -> None:
        """Persist all pipeline outputs — responses JSON and evaluation report.

        Args:
            all_convs: Conversations with llm_answers populated.
            accuracy: Average accuracy across all evaluated conversations.
        """
        self._save_responses(all_convs)
        self._save_evaluation(accuracy)

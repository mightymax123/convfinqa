"""
Parses ConvFinQA data from a JSON file into typed ConvQA objects.
"""

import json
import os
from typing import Any, cast

from loguru import logger

from app.models import ConvQA, FinancialDoc


class ConvFinQaDataParser:
    """Parses the ConvFinQA JSON dataset into a list of ConvQA instances."""

    def __init__(self, data_path: str, load_train_data: bool) -> None:
        """Initialise the parser and load raw JSON from disk.

        Args:
            data_path: Path to the ConvFinQA JSON dataset file.
            load_train_data: If True, use the training split; otherwise use the dev split.
        """
        self.data = self._load_json(data_path)
        self.split = "train" if load_train_data else "dev"

    def _load_json(self, data_path: str) -> dict[str, Any]:
        """Load JSON data from a file.

        Args:
            data_path: The path to the JSON file.

        Returns:
            The loaded JSON data as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid JSON or does not have a .json extension.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        if not data_path.endswith(".json"):
            raise ValueError("The provided file is not a JSON file. Please provide a valid JSON file.")

        try:
            with open(data_path, encoding="utf-8") as file:
                logger.info(f"Loading data from {data_path}")
                return cast(dict[str, Any], json.load(file))
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from the file {data_path}: {e}") from e

    def _parse_questions_and_answers(self, idx: int) -> tuple[list[str], list[str]]:
        """Parse questions and answers from the dialogue at the given index.

        Args:
            idx: The index of the entry.

        Returns:
            A tuple of (questions, answers) lists.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        questions = self.data[self.split][idx]["dialogue"]["conv_questions"]
        answers = self.data[self.split][idx]["dialogue"]["conv_answers"]
        return questions, answers

    def _parse_document(self, idx: int) -> FinancialDoc:
        """Parse the financial document at the given index.

        Args:
            idx: The index of the entry.

        Returns:
            The structured financial document.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return FinancialDoc.model_validate(self.data[self.split][idx]["doc"])

    def _parse_document_id(self, idx: int) -> str:
        """Parse the document ID at the given index.

        Args:
            idx: The index of the entry.

        Returns:
            The document ID string.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return cast(str, self.data[self.split][idx]["id"])

    def _parse_conversation(self, idx: int) -> ConvQA:
        """Parse a single conversation entry at the given index.

        Args:
            idx: The index of the entry.

        Returns:
            A ConvQA instance containing the document, questions, and answers.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        id = self._parse_document_id(idx)
        doc = self._parse_document(idx)
        questions, answers = self._parse_questions_and_answers(idx)
        return ConvQA(id=id, doc=doc, questions=questions, answers=answers)

    def parse_all_conversations(self) -> list[ConvQA]:
        """Parse all conversation entries from the selected split.

        Returns:
            A list of ConvQA instances for every entry in the split.
        """
        return [self._parse_conversation(idx) for idx in range(len(self.data[self.split]))]

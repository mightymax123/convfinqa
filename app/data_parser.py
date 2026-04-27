"""
Parses ConvFinQa data from a JSON file and provides methods to access question-answer pairs and documents.
"""

import json
import os
from typing import Any, Self, cast

from loguru import logger
from pydantic import BaseModel, Field, model_validator


class FinancialDoc(BaseModel):
    """Structured representation of a financial document with text and tabular data."""

    pre_text: str = Field(description="Introductory text preceding the table.")
    post_text: str = Field(description="Explanatory text following the table.")
    table: dict[str, dict[str, float | str | None]] = Field(description="Tabular data keyed by column then row label.")

    def _format_table(self) -> str:
        """Render the table dict as a plain-text grid.

        Returns:
            A tab-separated grid with column headers and row labels.
        """
        if not self.table:
            return ""

        columns = list(self.table.keys())
        row_labels = list(next(iter(self.table.values())).keys())

        header = "\t".join([""] + columns)
        rows = ["\t".join([label] + [str(self.table[col].get(label, "")) for col in columns]) for label in row_labels]
        return "\n".join([header] + rows)

    @property
    def formatted_doc(self) -> str:
        """Render the document as a readable string for use in prompts.

        Returns:
            A formatted string combining pre_text, the table as a grid, and post_text.
        """
        return "\n\n".join(
            [
                self.pre_text,
                self._format_table(),
                self.post_text,
            ]
        )


class ConvQA(BaseModel):
    """
    Class to represent a conversation question-answer pair.

    Provides validation and formatting for financial conversation data.
    """

    id: str = Field(min_length=1, description="Unique identifier for the conversation")
    doc: FinancialDoc = Field(description="The structured financial document related to the conversation")
    questions: list[str] = Field(min_length=1, description="List of questions in the conversation")
    answers: list[str] = Field(min_length=1, description="List of answers for the conversation")
    llm_answers: list[str] = Field(default_factory=list, description="Structured answers returned by the LLM.")

    @property
    def formatted_questions(self) -> str:
        """Format questions with delimiter for prompt generation."""
        return " {next_question} ".join(self.questions)

    @model_validator(mode="after")
    def validate_questions_and_answers_same_length(self) -> Self:
        """Validate that questions and answers lists have the same length."""
        if len(self.questions) != len(self.answers):
            raise ValueError(
                f"Document {self.id}: Questions and answers must have the same length. "
                f"Got {len(self.questions)} questions and {len(self.answers)} answers."
            )
        return self


class ConvFinQaDataParser:
    """
    A class to parse ConvFinQa data from a JSON file.
    """

    def __init__(self, data_path: str, load_train_data: bool = True) -> None:
        self.data = self._load_json(data_path)
        self.split = "train" if load_train_data else "dev"

    def _load_json(self, data_path: str) -> dict[str, Any]:
        """
        Load JSON data from a file.

        Args:
            data_path (str): The path to the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a valid JSON file or if there is an error decoding the JSON.

        Returns:
            dict[str, Any]: The loaded JSON data as a dictionary.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The file {data_path} does not exist.")

        if not data_path.endswith(".json"):
            raise ValueError("The provided file is not a JSON file. Please provide a valid JSON file.")

        try:
            with open(data_path, encoding="utf-8") as file:
                logger.info(f"Loading data from {data_path}")
                data = cast(dict[str, Any], json.load(file))
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from the file {data_path}: {e}") from e

    def _parse_questions_and_answers(self, idx: int) -> tuple[list[str], list[str]]:
        """
        Parse questions and answers from the dialogue at the given index.

        Args:
            idx (int): The index of the entry.

        Returns:
            tuple[list[str], list[str]]: A tuple containing a list of questions and a list of answers.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        logger.debug(f"Fetching Q&A pair at index {idx} from {self.split} split.")

        questions = self.data[self.split][idx]["dialogue"]["conv_questions"]
        answers = self.data[self.split][idx]["dialogue"]["conv_answers"]

        return questions, answers

    def _parse_document(self, idx: int) -> FinancialDoc:
        """
        Parse the financial document at the given index.

        Args:
            idx (int): The index of the entry.

        Returns:
            FinancialDoc: The structured financial document.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return FinancialDoc.model_validate(self.data[self.split][idx]["doc"])

    def _parse_document_id(self, idx: int) -> str:
        """
        Parse the document ID at the given index.

        Args:
            idx (int): The index of the entry.

        Returns:
            str: The document ID.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return cast(str, self.data[self.split][idx]["id"])

    def _parse_conversation(self, idx: int) -> ConvQA:
        """
        Parse a single conversation entry at the given index.

        Args:
            idx (int): The index of the entry.

        Returns:
            ConvQA: An instance of ConvQA containing the document, questions, and answers.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        logger.debug(f"Parsing conversation at index {idx} from {self.split} split.")

        id = self._parse_document_id(idx)
        doc = self._parse_document(idx)
        questions, answers = self._parse_questions_and_answers(idx)
        return ConvQA(id=id, doc=doc, questions=questions, answers=answers)

    def parse_all_conversations(self) -> list[ConvQA]:
        """
        Parse all conversation entries from the selected split.

        Returns:
            list[ConvQA]: A list of ConvQA instances containing all documents, questions, and answers.
        """
        return [self._parse_conversation(idx) for idx in range(len(self.data[self.split]))]

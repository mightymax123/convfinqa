"""
parses ConvFinQa data from a JSON file and provides methods to access question-answer pairs and documents.
"""

import json
import os
from typing import Any, cast

from pydantic import BaseModel, Field

from .logger import get_logger

logger = get_logger(__name__)


class ConvQA(BaseModel):
    """
    Class to represent a conversation question-answer pair.

    Provides validation and formatting for financial conversation data.
    """

    id: str = Field(min_length=1, description="Unique identifier for the conversation")
    doc: str = Field(min_length=1, description="The document text related to the conversation")
    questions: list[str] = Field(min_length=1, description="List of questions in the conversation")
    answers: list[str] = Field(min_length=1, description="List of answers for the conversation")
    llm_response: str | None = Field(default=None, description="Raw response from the language model")
    formatted_llm_response: list[str] = Field(default_factory=list, description="Parsed LLM response as list")

    @property
    def formatted_questions(self) -> str:
        """Format questions with delimiter for prompt generation."""
        return " {next_question} ".join(self.questions)

    def model_post_init(self, __context) -> None:
        """Validate that questions and answers lists have the same length."""
        if len(self.questions) != len(self.answers):
            raise ValueError(
                f"Document {self.id}: Questions and answers must have the same length. "
                f"Got {len(self.questions)} questions and {len(self.answers)} answers."
            )


class ConvFinQaDataParser:
    """
    A class to parse ConvFinQa data from a JSON file.
    """

    def __init__(self, data_path: str) -> None:
        self.data = self._load_json(data_path)

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

    def _get_q_and_a_pair(self, idx: int, data_type: str = "train") -> tuple[list[str], list[str]]:
        """
        Get a question and answer pair from the data by index.

        Args:
            idx (int): The index of the question-answer pair.
            data_type (str): The type of data to use ("train" or "dev").

        Returns:
            tuple[list[str], list[str]]: A tuple containing a list of questions and a list of answers.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        logger.debug(f"Fetching Q&A pair at index {idx} from {data_type} data.")

        questions = self.data[data_type][idx]["dialogue"]["conv_questions"]
        answers = self.data[data_type][idx]["dialogue"]["conv_answers"]

        return questions, answers

    def _get_doc_from_idx(self, idx: int, data_type: str = "train") -> str:
        """
        Get the document from the data by index.

        Args:
            idx (int): The index of the document.
            data_type (str): The type of data to use ("train" or "dev").

        Returns:
            str: The document text.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        doc = self.data[data_type][idx]["doc"]

        if isinstance(doc, dict):
            return str(doc)
        return cast(str, doc)

    def _get_doc_id_from_idx(self, idx: int, data_type: str = "train") -> str:
        """
        Get the document ID from the data by index.

        Args:
            idx (int): The index of the document.
            data_type (str): The type of data to use ("train" or "dev").

        Returns:
            str: The document ID.
        """
        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return cast(str, self.data[data_type][idx]["id"])

    def _get_doc_and_q_and_a_pair(self, idx: int, data_type: str = "train") -> ConvQA:
        """
        Get the document and a question-answer pair from the data by index.

        Args:
            idx (int): The index of the document and question-answer pair.
            data_type (str): The type of data to use ("train" or "dev").

        Returns:
            ConvQA: An instance of ConvQA containing the document, questions, and answers.
        """
        logger.debug(f"Fetching document and Q&A pair at index {idx} from {data_type} data.")

        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        id = self._get_doc_id_from_idx(idx, data_type)
        doc = self._get_doc_from_idx(idx, data_type)
        questions, answers = self._get_q_and_a_pair(idx, data_type)
        conv_qa = ConvQA(id=id, doc=doc, questions=questions, answers=answers)
        return conv_qa

    def get_all_docs_and_q_and_a_pairs(self, load_train_data: bool = True) -> list[ConvQA]:
        """
        Get all documents and question-answer pairs from the data.

        Args:
            load_train_data (bool): Whether to use training data or test data.

        Returns:
            list[ConvQA]: A list of ConvQA instances containing all documents, questions, and answers.
        """
        all_docs = []

        data_type = "train" if load_train_data else "dev"

        for idx in range(len(self.data[data_type])):
            conv_qa = self._get_doc_and_q_and_a_pair(idx, data_type)
            all_docs.append(conv_qa)

        return all_docs

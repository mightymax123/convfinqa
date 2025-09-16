"""
parses ConvFinQa data from a JSON file and provides methods to access question-answer pairs and documents.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConvQA:
    """
    Class to represent a conversation question-answer pair.

    Attributes:
        id (str): Unique identifier for the conversation question-answer pair.
        doc (str): The document text related to the question-answer pair.
        questions (str): The questions asked in the conversation.
        answers (str): The answers provided in the conversation.
        llm_response (str, optional): The response from the language model. Defaults to None.
    """

    id: str
    doc: str
    questions: list[str]
    answers: list[str]
    llm_response: Optional[str] = field(default=None)
    formatted_llm_response: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Post-initialization processing to get the formatted prompt from the questions.
        """
        self.formatted_questions = " {next_question} ".join(self.questions)


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
            raise ValueError(
                "The provided file is not a JSON file. Please provide a valid JSON file."
            )

        try:
            with open(data_path, encoding="utf-8") as file:
                logger.info(f"Loading data from {data_path}")
                data = cast(dict[str, Any], json.load(file))
                return data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from the file {data_path}: {e}"
            ) from e

    def _get_q_and_a_pair(
        self, idx: int, load_train_data: bool = True
    ) -> tuple[list[str], list[str]]:
        """
        Get a question and answer pair from the data by index.

        Args:
            idx (int): The index of the question-answer pair.
            load_train_data (bool): Whether to use training data or test data.

        Returns:
            tuple[list[str], list[str]]: A tuple containing a list of questions and a list of answers.
        """
        data_type = "train" if load_train_data else "dev"

        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        logger.debug(f"Fetching Q&A pair at index {idx} from {data_type} data.")

        questions = self.data[data_type][idx]["dialogue"]["conv_questions"]
        answers = self.data[data_type][idx]["dialogue"]["conv_answers"]

        return questions, answers

    def _get_doc_from_idx(self, idx: int, load_train_data: bool = True) -> str:
        """
        Get the document from the data by index.

        Args:
            idx (int): The index of the document.
            load_train_data (bool): Whether to use training data or test data.

        Returns:
            str: The document text.
        """
        data_type = "train" if load_train_data else "dev"

        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return cast(str, self.data[data_type][idx]["doc"])

    def _get_doc_id_from_idx(self, idx: int, load_train_data: bool = True) -> str:
        """
        Get the document ID from the data by index.

        Args:
            idx (int): The index of the document.
            load_train_data (bool): Whether to use training data or test data.

        Returns:
            str: The document ID.
        """
        data_type = "train" if load_train_data else "dev"

        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        return cast(str, self.data[data_type][idx]["id"])

    def _get_doc_and_q_and_a_pair(
        self, idx: int, load_train_data: bool = True
    ) -> ConvQA:
        """
        Get the document and a question-answer pair from the data by index.

        Args:
            idx (int): The index of the document and question-answer pair.
            load_train_data (bool): Whether to use training data or test data.

        Returns:
            ConvQA: An instance of ConvQA containing the document, questions, and answers.
        """
        logger.debug(
            f"Fetching document and Q&A pair at index {idx} from {'train' if load_train_data else 'dev'} data."
        )

        if idx < 0:
            raise ValueError("Index must be a non-negative integer.")

        id = self._get_doc_id_from_idx(idx, load_train_data)
        doc = self._get_doc_from_idx(idx, load_train_data)
        questions, answers = self._get_q_and_a_pair(idx, load_train_data)
        conv_qa = ConvQA(id=id, doc=doc, questions=questions, answers=answers)
        return conv_qa

    def get_all_docs_and_q_and_a_pairs(
        self, load_train_data: bool = True
    ) -> list[ConvQA]:
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
            conv_qa = self._get_doc_and_q_and_a_pair(idx, load_train_data)
            all_docs.append(conv_qa)

        return all_docs

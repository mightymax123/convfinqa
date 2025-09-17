"""
Generate LLM responses for conversations in the ConvFinQA dataset.
"""

import ast
import json
import os
import random
import re

from tqdm import tqdm

from src.config import config
from src.data_parser import ConvFinQaDataParser, ConvQA
from src.logger import get_logger
from src.model_loader import OpenAiLlmResponse, RetryConfig
from src.prompting import PromptGenerator

logger = get_logger(__name__)


class GetAllLlmResponses:
    def __init__(
        self,
        model_name: str = "gpt-4.1",
        prompting_strategy: str = "chain_of_thought",
        data_path: str | None = None,
        load_train_data: bool = False,
        sample_size: int = 100,
        use_seed: bool = True,
    ):
        """
        Initializes the GetAllLlmResponses class with the specified model name, prompting strategy, data path, and whether to load training data.
        
        Args:
            model_name (str): The name of the LLM model to use.
            prompting_strategy (str): The strategy for generating prompts.
            data_path (str | None): The path to the conversation dataset. If None, uses config default.
            load_train_data (bool): Whether to load training data or not. (default: False)
            sample_size (int): If specified, randomly sample this many conversations from the dataset.
            use_seed (bool): If True, sets a random seed for reproducibility. (default: True)
        """
        retry_config = RetryConfig(
            max_retries=config.max_retries,
            base_delay=config.base_delay
        )
        self.llm = OpenAiLlmResponse(
            model_name=model_name,
            retry_config=retry_config
        )

        actual_data_path = data_path if data_path is not None else config.data_path
        self.conv_parser = ConvFinQaDataParser(data_path=actual_data_path)

        self.all_convs = self.conv_parser.get_all_docs_and_q_and_a_pairs(load_train_data=load_train_data)
        self.prompt_gen = PromptGenerator(strategy=prompting_strategy)

        logger.info(
            f"Initializing GetAllLlmResponses with model: {model_name}, and prompting strategy: {prompting_strategy}"
        )

        if sample_size is not None:
            logger.info(f"sampling {sample_size} conversations from the dataset")
            if use_seed:
                logger.info(f"Using fixed random seed {config.random_seed} for reproducibility")
                random.seed(config.random_seed)
            self.all_convs = random.sample(self.all_convs, sample_size)

        subfolder = f"{model_name}_{prompting_strategy}"
        self.save_path = os.path.join("/app/outputs", subfolder, "convfinqa_responses.json")

    def _get_conv_response(self, conv: ConvQA) -> None:
        """
        Get the LLM response for a single conversation append the original and formatted responses to the conversation object.

        Args:
            conv (ConvQA): The conversation object containing questions and answers.
        """
        logger.debug(f"Generating prompt and requesting response for conversation ID: {conv.id}")

        prompt = self.prompt_gen.generate_prompt(conv)
        response = self.llm.get_response(prompt=prompt)
        conv.llm_response = response
        conv.formatted_llm_response = self._extract_list_from_llm_response(response)

        logger.debug(f"Response for conversation ID {conv.id} received and processed.")

    def _extract_list_from_llm_response(self, llm_response: str) -> list[str]:
        """
        Extracts the last list of strings from an LLM response (should only be 1 list but to cover edge cases).

        Args:
            llm_response (str): Full text response from the LLM.

        Returns:
            list[str]: The extracted list of strings, or an empty list if not found or invalid.
        """
        if not llm_response:
            logger.warning("Received empty LLM response.")
            return []

        matches = re.findall(r"\[[^\[\]]+\]", llm_response)
        if not matches:
            logger.warning("No valid list found in the LLM response.")
            return []

        last = matches[-1]
        try:
            result = ast.literal_eval(last)
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                return result
        except (SyntaxError, ValueError):
            pass

        return []

    def get_all_responses(self) -> list[ConvQA]:
        """
        Get LLM responses for all conversations in the dataset.
        """
        for conv in tqdm(self.all_convs, desc="Processing conversations", unit="conv"):
            try:
                self._get_conv_response(conv)
            except Exception as e:
                logger.error(f"Error processing conversation {conv.id}: {e}")
                raise RuntimeError(f"Error processing conversation {conv.id}: {e}") from e

        self._save_conversations_to_json()

        return self.all_convs

    def _save_conversations_to_json(self) -> None:
        """
        Save a list of ConvQA objects to a JSON file.


        raises:
            ValueError: If the list of conversations is empty.
        """
        if not self.all_convs:
            raise ValueError("The list of conversations is empty.")

        dir_path = os.path.dirname(self.save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        data = [
            {
                "id": conv.id,
                "doc": conv.doc,
                "questions": conv.questions,
                "answers": conv.answers,
                "formatted_llm_response": conv.formatted_llm_response,
            }
            for conv in self.all_convs
        ]
        logger.info(f"Saving {len(data)} conversations to {self.save_path}")

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversations saved successfully to {self.save_path}")

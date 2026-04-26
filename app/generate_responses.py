"""
Generate LLM responses for conversations in the ConvFinQA dataset.
"""

import ast
import json
import os
import random
import re

from loguru import logger
from tqdm import tqdm

from app.data_parser import ConvFinQaDataParser, ConvQA
from app.model_loader import ModelName, OpenAiLlmResponse, RetryConfig
from app.prompting import PromptGenerator, PromptingStrategy
from app.settings import get_settings


class GetAllLlmResponses:
    def __init__(
        self,
        model_name: ModelName = ModelName.GPT_4_1,
        prompting_strategy: PromptingStrategy = PromptingStrategy.CHAIN_OF_THOUGHT,
        load_train_data: bool = False,
        sample_size: int = 100,
        use_seed: bool = True,
    ):
        """
        Initialise with model, prompting strategy, and sampling options.

        Args:
            model_name: The LLM model to use.
            prompting_strategy: The strategy for generating prompts.
            load_train_data: Whether to load training data instead of the dev set.
            sample_size: Number of conversations to randomly sample from the dataset.
            use_seed: If True, sets a fixed random seed for reproducibility.
        """
        settings = get_settings()
        retry_config = RetryConfig(max_retries=settings.max_retries, base_delay=settings.base_delay)
        self.llm = OpenAiLlmResponse(model_name=model_name, retry_config=retry_config)

        self.conv_parser = ConvFinQaDataParser(data_path=settings.data_path, load_train_data=load_train_data)
        self.all_convs = self.conv_parser.parse_all_conversations()

        self.prompt_gen = PromptGenerator(strategy=prompting_strategy)

        logger.info(
            f"Initialising GetAllLlmResponses with model: {model_name.value}, and prompting strategy: {prompting_strategy.value}"
        )

        if sample_size is not None:
            logger.info(f"Sampling {sample_size} conversations from the dataset")
            if use_seed:
                logger.info(f"Using fixed random seed {settings.random_seed} for reproducibility")
                random.seed(settings.random_seed)
            self.all_convs = random.sample(self.all_convs, sample_size)

        subfolder = f"{model_name.value}_{prompting_strategy.value}"
        self.save_path = os.path.join("/code/outputs", subfolder, "convfinqa_responses.json")

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

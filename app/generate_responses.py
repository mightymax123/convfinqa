"""
Generate LLM responses for conversations in the ConvFinQA dataset.
"""

import json
import os
import random

from loguru import logger
from tqdm import tqdm

from app.agent import LlmAnswers, ModelName, build_agent, get_response
from app.data_parser import ConvFinQaDataParser, ConvQA
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

        self.agent = build_agent(model_name=model_name, max_retries=settings.max_retries)
        self.prompt_gen = PromptGenerator(strategy=prompting_strategy)

        conv_parser = ConvFinQaDataParser(data_path=settings.data_path, load_train_data=load_train_data)
        self.all_convs = conv_parser.parse_all_conversations()

        logger.info(
            f"Initialising GetAllLlmResponses with model: {model_name.value}, "
            f"and prompting strategy: {prompting_strategy.value}"
        )

        if sample_size is not None:
            logger.info(f"Sampling {sample_size} conversations from the dataset")
            if use_seed:
                logger.info(f"Using fixed random seed {settings.random_seed} for reproducibility")
                random.seed(settings.random_seed)
            self.all_convs = random.sample(self.all_convs, sample_size)

        subfolder = f"{model_name.value}_{prompting_strategy.value}"
        self.save_path = os.path.join("/code/outputs", subfolder, "convfinqa_responses.json")

    def _get_conv_response(self, conv: ConvQA) -> None:
        """Get the LLM response for a single conversation and store structured answers.

        Args:
            conv: The conversation object containing questions and answers.
        """
        logger.debug(f"Generating prompt and requesting response for conversation ID: {conv.id}")

        prompt = self.prompt_gen.generate_prompt(conv)
        llm_answers: LlmAnswers = get_response(self.agent, prompt)
        conv.llm_answers = llm_answers.answers

        logger.debug(f"Response for conversation ID {conv.id} received and processed.")

    def _save_conversations_to_json(self) -> None:
        """Save the list of conversations with LLM answers to a JSON file.

        Raises:
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
                "doc": conv.doc.model_dump(),
                "questions": conv.questions,
                "answers": conv.answers,
                "llm_answers": conv.llm_answers,
            }
            for conv in self.all_convs
        ]
        logger.info(f"Saving {len(data)} conversations to {self.save_path}")

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversations saved successfully to {self.save_path}")

    def get_all_responses(self) -> list[ConvQA]:
        """Get LLM responses for all conversations in the dataset.

        Returns:
            The list of conversations with llm_answers populated.

        Raises:
            RuntimeError: If any individual conversation fails to process.
        """
        for conv in tqdm(self.all_convs, desc="Processing conversations", unit="conv"):
            try:
                self._get_conv_response(conv)
            except Exception as e:
                logger.error(f"Error processing conversation {conv.id}: {e}")
                raise RuntimeError(f"Error processing conversation {conv.id}: {e}") from e

        self._save_conversations_to_json()

        return self.all_convs

"""
Generate LLM responses for conversations in the ConvFinQA dataset.
"""

import random

from loguru import logger
from tqdm import tqdm

from app.agent import LlmAnswers, ModelName, build_agent, get_response
from app.data_parser import ConvFinQaDataParser, ConvQA
from app.prompting import PromptGenerator, PromptingStrategy
from app.settings import get_settings

DATA_PATH = "/data/convfinqa_dataset.json"


class GetAllLlmResponses:
    def __init__(
        self,
        model_name: ModelName,
        prompting_strategy: PromptingStrategy,
        load_train_data: bool,
        sample_size: int,
        seed: int | None,
    ):
        """
        Initialise with model, prompting strategy, and sampling options.

        Args:
            model_name: The LLM model to use.
            prompting_strategy: The strategy for generating prompts.
            load_train_data: Whether to load training data instead of the dev set.
            sample_size: Number of conversations to randomly sample from the dataset.
            seed: Random seed for reproducible sampling. If None, sampling is non-deterministic.
        """
        settings = get_settings()

        self.max_retries = settings.max_retries
        self.agent = build_agent(
            model_name=model_name,
            max_retries=settings.max_retries,
        )
        self.prompt_gen = PromptGenerator(strategy=prompting_strategy)

        conv_parser = ConvFinQaDataParser(data_path=DATA_PATH, load_train_data=load_train_data)
        self.all_convs = conv_parser.parse_all_conversations()

        logger.info(
            f"Initialising GetAllLlmResponses with model: {model_name.value}, "
            f"and prompting strategy: {prompting_strategy.value}"
        )

        if sample_size is not None:
            logger.info(f"Sampling {sample_size} conversations from the dataset")
            if seed is not None:
                logger.info(f"Using fixed random seed {seed} for reproducibility")
                random.seed(seed)
            self.all_convs = random.sample(self.all_convs, sample_size)

    async def _get_conv_response(self, conv: ConvQA) -> None:
        """Get the LLM response for a single conversation and store structured answers.

        If the agent exceeds the request limit, get_response returns None and
        the conversation is skipped with an empty llm_answers list. All other
        exceptions bubble up to the caller.

        Args:
            conv: The conversation object containing questions and answers.
        """
        logger.debug(f"\n--- Conversation: {conv.id} ---")

        prompt = self.prompt_gen.generate_prompt(conv)
        llm_answers: LlmAnswers | None = await get_response(self.agent, prompt, max_retries=self.max_retries)

        if llm_answers is None:
            logger.warning(
                f"Skipping conversation {conv.id}: request limit exceeded. "
                "All answers for this conversation will be empty and scored as incorrect."
            )
            return

        conv.llm_answers = llm_answers.answers
        logger.debug(f"Conversation {conv.id} complete — answers: {llm_answers.answers}")

    async def get_all_responses(self) -> list[ConvQA]:
        """Get LLM responses for all conversations in the dataset sequentially.

        Each conversation is fully processed before the next begins, avoiding
        thundering-herd rate-limit issues when calling the OpenAI API.

        Returns:
            The list of conversations with llm_answers populated.

        Raises:
            RuntimeError: If any individual conversation fails to process.
        """
        for conv in tqdm(self.all_convs, desc="Processing conversations", unit="conv"):
            await self._get_conv_response(conv)

        return self.all_convs

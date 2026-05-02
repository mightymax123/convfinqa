from abc import ABC, abstractmethod
from enum import Enum

from loguru import logger

from app.data_parser import ConvQA


class PromptingStrategy(str, Enum):
    """Enum for supported prompting strategies."""

    BASIC = "basic"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"


class PromptStrategy(ABC):
    @abstractmethod
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Generate a prompt based on the document and questions.
        Args:
            doc (str): The document containing relevant information.
            questions (str): The formatted questions to be answered.
        Returns:
            str: The generated prompt string.
        """
        pass


class BasicPromptStrategy(PromptStrategy):
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Construct a minimal prompt containing only the document and questions.

        The system prompt already provides all task framing and tool-use instructions.

        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.

        Returns:
            str: The generated prompt string.
        """
        return f"Document:\n{doc}\n\nQuestions:\n{questions}"


class ChainOfThoughtPromptStrategy(PromptStrategy):
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Construct a prompt that prepends a step-by-step reasoning instruction.

        The system prompt handles task framing; this adds only the CoT nudge.
        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.
        Returns:
            str: The generated prompt string.
        """
        return (
            "Think through each question step-by-step before arriving at a final numeric answer.\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}"
        )


class FewShotPromptStrategy(PromptStrategy):
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Construct a few-shot prompt with example Q&A pairs drawn from the dataset.

        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.

        Returns:
            str: The generated prompt string.
        """
        return (
            "Here are three example Q&A pairs:\n\n"
            "Questions:\n"
            "what was the weighted average exercise price per share in 2007? {next_question} "
            "and what was it in 2005? {next_question} "
            "what was, then, the change over the years? {next_question} "
            "what was the weighted average exercise price per share in 2005? {next_question} "
            "and how much does that change represent in relation to this 2005 weighted average exercise price?\n"
            "Answers: 60.94 | 25.14 | 35.80 | 25.14 | 142.4%\n\n"
            "Questions:\n"
            "what was the change in the unamortized debt issuance costs associated with the senior notes between 2016 and 2017? {next_question} "
            "so what was the percentage change during this time? {next_question} "
            "what was the change associated with credit facilities during that time? {next_question} "
            "so what was the percentage change?\n"
            "Answers: -4 | -21.1% | 3 | 37.5%\n\n"
            "Questions:\n"
            "what is the ratio of discretionary company contributions to total expensed amounts for savings plans in 2009? {next_question} "
            "what is that times 100?\n"
            "Answers: 0.1083 | 10.83\n\n"
            "Now answer the following.\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}"
        )


class PromptGenerator:
    _STRATEGY_DICT: dict[PromptingStrategy, type[PromptStrategy]] = {
        PromptingStrategy.BASIC: BasicPromptStrategy,
        PromptingStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtPromptStrategy,
        PromptingStrategy.FEW_SHOT: FewShotPromptStrategy,
    }

    def __init__(self, strategy: PromptingStrategy) -> None:
        """
        Initialise the PromptGenerator with a specific strategy.

        Args:
            strategy: The prompting strategy to use.
        """
        self._strategy = self._STRATEGY_DICT[strategy]()
        logger.info(f"Using prompt strategy: {strategy.value}")

    def generate_prompt(self, conversation: ConvQA) -> str:
        """
        Generate a prompt using the specified strategy, given a document and questions.

        Args:
            conversation (ConvQA): The conversation object containing document and questions.

        Returns:
            str: The generated prompt string.
        """
        doc = conversation.doc.formatted_doc
        questions = conversation.formatted_questions

        return self._strategy.generate_prompt(doc, questions)

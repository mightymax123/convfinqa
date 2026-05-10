"""
Prompt generation strategies for the ConvFinQA pipeline.

Implements the Strategy pattern: each PromptingStrategy enum value maps to a
concrete PromptStrategy subclass. Adding a new strategy requires only a new
enum member, a new subclass, and one entry in PromptGenerator._STRATEGY_DICT.
"""

from abc import ABC, abstractmethod

from loguru import logger

from app.models import ConvQA, PromptingStrategy


class PromptStrategy(ABC):
    """Abstract base for prompt generation strategies."""

    @abstractmethod
    def generate_prompt(self, doc: str, questions: str) -> str:
        """Generate a prompt from the formatted document and questions.

        Args:
            doc: The rendered financial document string.
            questions: The formatted questions string with {next_question} delimiters.

        Returns:
            The complete prompt string to send to the LLM.
        """


class BasicPromptStrategy(PromptStrategy):
    """Minimal prompt containing only the document and questions."""

    def generate_prompt(self, doc: str, questions: str) -> str:
        """Construct a minimal prompt containing only the document and questions.

        The system prompt already provides all task framing and tool-use instructions.

        Args:
            doc: The financial document with 'pre_text', 'post_text', and 'table'.
            questions: A formatted string with {next_question} as delimiters.

        Returns:
            The generated prompt string.
        """
        return f"Document:\n{doc}\n\nQuestions:\n{questions}"


class ChainOfThoughtPromptStrategy(PromptStrategy):
    """Prompt that prepends a step-by-step reasoning instruction."""

    def generate_prompt(self, doc: str, questions: str) -> str:
        """Construct a prompt that prepends a step-by-step reasoning instruction.

        The system prompt handles task framing; this adds only the CoT nudge.

        Args:
            doc: The financial document with 'pre_text', 'post_text', and 'table'.
            questions: A formatted string with {next_question} as delimiters.

        Returns:
            The generated prompt string.
        """
        return (
            "Think through each question step-by-step before arriving at a final numeric answer.\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}"
        )


class FewShotPromptStrategy(PromptStrategy):
    """Prompt with example Q&A pairs drawn from the dataset."""

    def generate_prompt(self, doc: str, questions: str) -> str:
        """Construct a few-shot prompt with example Q&A pairs drawn from the dataset.

        Args:
            doc: The financial document with 'pre_text', 'post_text', and 'table'.
            questions: A formatted string with {next_question} as delimiters.

        Returns:
            The generated prompt string.
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
    """Selects a prompting strategy and generates prompts from ConvQA objects."""

    _STRATEGY_DICT: dict[PromptingStrategy, type[PromptStrategy]] = {
        PromptingStrategy.BASIC: BasicPromptStrategy,
        PromptingStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtPromptStrategy,
        PromptingStrategy.FEW_SHOT: FewShotPromptStrategy,
    }

    def __init__(self, strategy: PromptingStrategy) -> None:
        """Initialise the PromptGenerator with a specific strategy.

        Args:
            strategy: The prompting strategy to use.
        """
        self._strategy = self._STRATEGY_DICT[strategy]()
        logger.info(f"Using prompt strategy: {strategy.value}")

    def generate_prompt(self, conversation: ConvQA) -> str:
        """Generate a prompt from a conversation using the configured strategy.

        Args:
            conversation: The conversation object containing document and questions.

        Returns:
            The generated prompt string.
        """
        doc = conversation.doc.formatted_doc
        questions = conversation.formatted_questions
        return self._strategy.generate_prompt(doc, questions)

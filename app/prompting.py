from abc import ABC, abstractmethod

from src.data_parser import ConvQA
from src.logger import get_logger

logger = get_logger(__name__)


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
        Construct a prompt by injecting doc and questions directly.

        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.

        Returns:
            str: The generated prompt string.
        """
        return (
            "You are a financial question answering assistant.\n"
            "You will be given a document containing financial context, including text and tables.\n"
            "You will also receive a series of questions separated by the token {next_question}.\n"
            "Answer all questions in order.\n"
            "Return only the **final numeric answers** (no full sentences), as a Python list of strings.\n"
            "For example: ['60.94', '25.14', '35.80', '25.14', '142.4%']\n"
            'Do not include any explanation or units like "dollars" or "%".\n'
            "Just return the values in order, as shown.\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}\n\n"
            "Answers (as a Python list of strings):"
        )


class ChainOfThoughtPromptStrategy(PromptStrategy):
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Construct a prompt that encourages step-by-step reasoning before providing final answers.
        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.
        Returns:
            str: The generated prompt string.
        """
        return (
            "You are a financial assistant trained to answer multi-step questions based on financial reports.\n"
            "You will be given a document containing financial context, including textual discussion and tabular data.\n"
            "You will also receive a series of questions separated by the token {next_question}.\n"
            "Please think through each question step-by-step before arriving at a final numeric answer.\n"
            "Then, return only the **final numeric answers**, including symbols like '£' or '%' if they are appropriate.\n"
            "Do not return explanations or full sentences — just the final values.\n"
            "Return the answers as a Python list of strings.\n\n"
            "Example format:\n"
            "Step-by-step reasoning: ...\n"
            "Final answers: ['£12.50', '25.14', '18%', '100']\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}\n\n"
            "Your reasoning and final answers:"
        )


class FewShotPromptStrategy(PromptStrategy):
    def generate_prompt(self, doc: str, questions: str) -> str:
        """
        Construct a few-shot prompt with strict emphasis on output formatting.

        Args:
            doc (str): The financial document with 'pre_text', 'post_text', and 'table'.
            questions (str): A formatted string with {next_question} as delimiters.

        Returns:
            str: The generated prompt string.
        """
        return (
            "You are a financial assistant. Your task is to answer a series of related questions given a document.\n"
            "Your answers must strictly follow this format:\n"
            "**Return ONLY the final numeric answers** as a Python list of strings — e.g., ['12.34', '56%', '78'].\n"
            "Do NOT include units (unless explicitly requested), explanations, or any text — just the raw values.\n"
            "The questions will be separated by the token {next_question}.\n\n"
            "Here are three example Q&A pairs:\n\n"
            "Questions:\n"
            "what was the weighted average exercise price per share in 2007? {next_question} "
            "and what was it in 2005? {next_question} "
            "what was, then, the change over the years? {next_question} "
            "what was the weighted average exercise price per share in 2005? {next_question} "
            "and how much does that change represent in relation to this 2005 weighted average exercise price?\n"
            "Answers:\n"
            "['60.94', '25.14', '35.80', '25.14', '142.4%']\n\n"
            "Questions:\n"
            "what was the change in the unamortized debt issuance costs associated with the senior notes between 2016 and 2017? {next_question} "
            "so what was the percentage change during this time? {next_question} "
            "what was the change associated with credit facilities during that time? {next_question} "
            "so what was the percentage change?\n"
            "Answers:\n"
            "['-4', '-21.1%', '3', '37.5%']\n\n"
            "Questions:\n"
            "what is the ratio of discretionary company contributions to total expensed amounts for savings plans in 2009? {next_question} "
            "what is that times 100?\n"
            "Answers:\n"
            "['0.1083', '10.83']\n\n"
            "Now answer the following using the same format.\n\n"
            f"Document:\n{doc}\n\n"
            f"Questions:\n{questions}\n\n"
            "Answers:"
        )


class PromptGenerator:
    _STRATEGY_DICT: dict[str, type[PromptStrategy]] = {
        "basic": BasicPromptStrategy,
        "chain_of_thought": ChainOfThoughtPromptStrategy,
        "few_shot": FewShotPromptStrategy,
    }

    def __init__(self, strategy: str = "basic") -> None:
        """
        Initialize the PromptGenerator with a specific strategy.

        Args:
            strategy (str): The strategy to use for generating prompts, comes from the stratergy dict. Defaults to 'basic'.
        """
        try:
            self._strategy = self._STRATEGY_DICT[strategy]()
            logger.info(f"Using prompt strategy: {strategy}")
        except KeyError as err:
            logger.error(f"Invalid prompting strategy '{strategy}' provided.")
            raise ValueError(
                f"Strategy '{strategy}' is not recognized. Available strategies: {list(self._STRATEGY_DICT.keys())}"
            ) from err

    def generate_prompt(self, conversation: ConvQA) -> str:
        """
        Generate a prompt using the specified strategy, given a document and questions.

        Args:
            conversation (ConvQA): The conversation object containing document and questions.

        Returns:
            str: The generated prompt string.
        """
        logger.debug(f"Generated prompt for conversation {conversation.id}")

        doc = conversation.doc
        questions = conversation.formatted_questions

        return self._strategy.generate_prompt(doc, questions)

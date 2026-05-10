"""
Core domain types and enumerations for the ConvFinQA pipeline.

This module is the innermost dependency — it imports only from stdlib and pydantic.
No other app module should be imported here.
"""

from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class ModelName(str, Enum):
    """Supported model names using OpenRouter provider-prefixed identifiers."""

    GPT_5_4_MINI = "openai/gpt-5.4-mini"
    GPT_5_5 = "openai/gpt-5.5"
    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4.5"
    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4.5"
    CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4.6"
    GEMINI_3_1_PRO = "google/gemini-3.1-pro-preview"
    GEMINI_3_1_FLASH_LITE = "google/gemini-3.1-flash-lite-preview"


class PromptingStrategy(str, Enum):
    """Supported prompting strategies."""

    BASIC = "basic"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"


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
    """A single conversational QA entry from the ConvFinQA dataset."""

    id: str = Field(min_length=1, description="Unique identifier for the conversation.")
    doc: FinancialDoc = Field(description="The structured financial document related to the conversation.")
    questions: list[str] = Field(min_length=1, description="List of questions in the conversation.")
    answers: list[str] = Field(min_length=1, description="List of ground-truth answers for the conversation.")
    llm_answers: list[str] = Field(default_factory=list, description="Structured answers returned by the LLM.")
    judge_verdicts: list[bool] = Field(
        default_factory=list, description="Per-answer correctness verdicts from the LLM judge."
    )

    @property
    def formatted_questions(self) -> str:
        """Format questions with delimiter for prompt generation.

        Returns:
            Questions joined by the {next_question} token used in prompts.
        """
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

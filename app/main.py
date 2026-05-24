"""
CLI entry point for the ConvFinQA evaluation pipeline.
"""

import asyncio
from typing import Optional

import typer
from pydantic import BaseModel
from rich import print as rich_print
from rich.pretty import Pretty

from app.agent import build_agent
from app.evaluator import ConversationsEvaluator
from app.generate_responses import GetAllLlmResponses
from app.judge import build_judge_agent
from app.log import configure_logging
from app.models import ModelName, PromptingStrategy
from app.results_writer import ResultsWriter

app = typer.Typer(
    name="convfinqa",
    help="app for ConvFinQA dataset evaluation",
    add_completion=True,
    no_args_is_help=True,
)


class MainArgs(BaseModel):
    """Validated arguments for the ConvFinQA pipeline."""

    model_name: ModelName
    prompting_strategy: PromptingStrategy
    sample_size: int
    use_train_data: bool
    seed: int | None


def main(args: MainArgs) -> None:
    """Run the ConvFinQA pipeline end-to-end.

    Orchestrates data ingestion, LLM response generation, LLM-judge evaluation,
    and output persistence in sequence.

    Args:
        args: Validated pipeline arguments.
    """
    agent = build_agent(model_name=args.model_name)
    generator = GetAllLlmResponses(
        agent=agent,
        prompting_strategy=args.prompting_strategy,
        sample_size=args.sample_size,
        load_train_data=args.use_train_data,
        seed=args.seed,
    )
    all_convs = asyncio.run(generator.get_all_responses())

    judge_agent = build_judge_agent()
    evaluator = ConversationsEvaluator(
        all_convs=all_convs,
        judge_agent=judge_agent,
    )
    accuracy = asyncio.run(evaluator.evaluate_all_conversations())

    writer = ResultsWriter(
        model_name=args.model_name,
        prompting_strategy=args.prompting_strategy,
        sample_size=args.sample_size,
    )
    writer.save_outputs(all_convs, accuracy)

    rich_print(f"[bold green]Average accuracy: {accuracy:.2f}%[/bold green]")


@app.command()
def evaluate(
    model_name: ModelName = typer.Option(ModelName.GEMINI_3_1_FLASH_LITE, help="Name of the LLM model to use"),  # noqa: B008
    prompting_strategy: PromptingStrategy = typer.Option(  # noqa: B008
        PromptingStrategy.BASIC,
        help="Prompting strategy to use",
    ),
    sample_size: int = typer.Option(10, help="Number of samples to evaluate"),
    use_train_data: bool = typer.Option(False, help="Use training data instead of dev set"),
    seed: Optional[int] = typer.Option(  # noqa: B008, UP007
        None, help="Random seed for reproducible sampling. Omit for non-deterministic sampling."
    ),
) -> None:
    """Run the ConvFinQA pipeline with specified parameters.

    Args:
        model_name: The LLM model to use.
        prompting_strategy: Prompting strategy to use.
        sample_size: Number of samples to evaluate.
        use_train_data: Whether to use training data instead of dev set.
        seed: Random seed for reproducible sampling. If omitted, sampling is non-deterministic.
    """
    args = MainArgs(
        model_name=model_name,
        prompting_strategy=prompting_strategy,
        sample_size=sample_size,
        use_train_data=use_train_data,
        seed=seed,
    )

    configure_logging()
    rich_print("[green]Running ConvFinQA with the following parameters:[/green]")
    rich_print(Pretty(args, expand_all=True))

    main(args)


if __name__ == "__main__":
    app()

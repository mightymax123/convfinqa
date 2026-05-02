"""
Main typer app for ConvFinQA
"""

import asyncio
from datetime import UTC, datetime

import typer
from loguru import logger
from pydantic import BaseModel, Field
from rich import print as rich_print
from rich.pretty import Pretty

from app.agent import ModelName
from app.evaluator import ConversationsEvaluator
from app.generate_responses import GetAllLlmResponses
from app.prompting import PromptingStrategy

LOG_FILE = "/code/logs/convfinqa.log"

app = typer.Typer(
    name="convfinqa",
    help="app for ConvFinQA dataset evaluation",
    add_completion=True,
    no_args_is_help=True,
)


def _configure_logging() -> None:
    """Add a file sink to loguru using the default log path from settings.

    The default stderr sink is kept. The file sink rotates at 50 MB and
    retains the last 5 files so logs do not grow without bound. A separator
    line is written to the file at the start of each run so individual runs
    are clearly partitioned when reviewing the log.
    """
    log_file = LOG_FILE
    logger.add(
        log_file,
        level="DEBUG",
        rotation="50 MB",
        retention=5,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}",
    )
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n  NEW RUN — {timestamp}\n{'=' * 60}\n\n")
    logger.info(f"Logging to file: {log_file}")


class MainArgs(BaseModel):
    """Validated arguments for the ConvFinQA pipeline."""

    model_name: ModelName = ModelName.GPT_4_1
    prompting_strategy: PromptingStrategy = PromptingStrategy.CHAIN_OF_THOUGHT
    sample_size: int = Field(default=10, gt=0)
    use_train_data: bool = False
    seed: int | None = None


def main(args: MainArgs) -> None:
    """
    Main function to run the ConvFinQA pipeline.

    Args:
        args: Validated pipeline arguments.
    """
    generator = GetAllLlmResponses(
        model_name=args.model_name,
        prompting_strategy=args.prompting_strategy,
        sample_size=args.sample_size,
        load_train_data=args.use_train_data,
        seed=args.seed,
    )
    all_convs = asyncio.run(generator.get_all_responses())

    evaluator = ConversationsEvaluator(
        all_convs=all_convs,
        model_name=args.model_name,
        prompting_strategy=args.prompting_strategy,
        sample_size=args.sample_size,
    )
    accuracy = evaluator.evaluate_all_conversations()

    rich_print(f"[bold green]Average accuracy: {accuracy:.2f}%[/bold green]")


@app.command()
def evaluate(
    model_name: ModelName = typer.Option(ModelName.GPT_4_1, help="Name of the LLM model to use"),  # noqa: B008
    prompting_strategy: PromptingStrategy = typer.Option(  # noqa: B008
        PromptingStrategy.CHAIN_OF_THOUGHT,
        help="Prompting strategy to use",
    ),
    sample_size: int = typer.Option(10, help="Number of samples to evaluate"),
    use_train_data: bool = typer.Option(False, help="Use training data instead of dev set"),
    seed: int | None = typer.Option(
        None, help="Random seed for reproducible sampling. Omit for non-deterministic sampling."
    ),  # noqa: B008
) -> None:
    """
    Run the ConvFinQA pipeline with specified parameters.

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

    _configure_logging()
    rich_print("[green]Running ConvFinQA with the following parameters:[/green]")
    rich_print(Pretty(args, expand_all=True))

    main(args)


if __name__ == "__main__":
    app()

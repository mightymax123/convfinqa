"""
Main typer app for ConvFinQA
"""

import asyncio

import typer
from pydantic import BaseModel, Field
from rich import print as rich_print
from rich.pretty import Pretty

from app.agent import ModelName
from app.evaluator import ConversationsEvaluator
from app.generate_responses import GetAllLlmResponses
from app.prompting import PromptingStrategy

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
    sample_size: int = Field(gt=0)
    use_train_data: bool
    use_seed: bool


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
        use_seed=args.use_seed,
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
    use_seed: bool = typer.Option(
        True,
        help="Use fixed random seed for reproducibility",
        is_flag=False,
    ),
) -> None:
    """
    Run the ConvFinQA pipeline with specified parameters.

    Args:
        model_name: The LLM model to use.
        prompting_strategy: Prompting strategy to use.
        sample_size (int): Number of samples to evaluate.
        use_train_data (bool): Whether to use training data instead of dev set.
        use_seed (bool): Whether to use a fixed random seed for reproducibility.
    """
    args = MainArgs(
        model_name=model_name,
        prompting_strategy=prompting_strategy,
        sample_size=sample_size,
        use_train_data=use_train_data,
        use_seed=use_seed,
    )

    rich_print("[green]Running ConvFinQA with the following parameters:[/green]")
    rich_print(Pretty(args, expand_all=True))

    main(args)


if __name__ == "__main__":
    app()

"""
Main typer app for ConvFinQA
"""

from typing import TypedDict

import typer
from rich import print as rich_print
from rich.pretty import Pretty

from src.evaluator import ConversationsEvaluator
from src.generate_responses import GetAllLlmResponses


class MainArgs(TypedDict):
    model_name: str
    prompting_strategy: str
    sample_size: int
    use_train_data: bool
    use_seed: bool


app = typer.Typer(
    name="main",
    help="app for ConvFinQA dataset evaluation",
    add_completion=True,
    no_args_is_help=True,
)


def main(args: MainArgs) -> None:
    """
    Main function to run the ConvFinQA pipeline.

    Args:
        args (dict): Dictionary containing parameters for the ConvFinQA pipeline.
    """
    generator = GetAllLlmResponses(
        model_name=args["model_name"],
        prompting_strategy=args["prompting_strategy"],
        sample_size=args["sample_size"],
        load_train_data=args["use_train_data"],
        use_seed=args["use_seed"],
    )
    all_convs = generator.get_all_responses()

    evaluator = ConversationsEvaluator(
        all_convs=all_convs,
        model_name=args["model_name"],
        prompting_strategy=args["prompting_strategy"],
        sample_size=args["sample_size"],
    )
    accuracy = evaluator.evaluate_all_conversations()

    rich_print(f"[bold green]Average accuracy: {accuracy:.2f}%[/bold green]")


@app.command()
def evaluate(
    model_name: str = typer.Option("gpt-4.1", help="Name of the LLM model to use"),
    prompting_strategy: str = typer.Option(
        "chain_of_thought",
        help="Prompting strategy to use (e.g. 'basic', 'chain_of_thought')",
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
        model_name (str): Name of the LLM model to use.
        prompting_strategy (str): Prompting strategy to use (e.g. 'basic', 'chain_of_thought').
        sample_size (int): Number of samples to evaluate.
        use_train_data (bool): Whether to use training data instead of dev set.
        use_seed (bool): Whether to use a fixed random seed for reproducibility.
    """
    args: MainArgs = {
        "model_name": model_name,
        "prompting_strategy": prompting_strategy,
        "sample_size": sample_size,
        "use_train_data": use_train_data,
        "use_seed": use_seed,
    }

    rich_print("[green]Running ConvFinQA with the following parameters:[/green]")
    rich_print(Pretty(args, expand_all=True))

    main(args)


if __name__ == "__main__":
    app()

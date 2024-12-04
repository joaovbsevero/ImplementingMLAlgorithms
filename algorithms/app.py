import typer

from .utils import get_algorithms, import_all_modules

app = typer.Typer()

import_all_modules("algorithms")
ALGORITHMS = get_algorithms()


@app.command()
def run(algorithm_name: str):
    """
    Run a specified algorithm.

    Args:
        algorithm_name (str): The name of the algorithm to run.
    """
    if algorithm_name in ALGORITHMS:
        metadata = ALGORITHMS[algorithm_name]
        metadata.func()
    else:
        typer.echo(
            f"Algorithm '{algorithm_name}' not found. "
            "Run `python -m algorithms list` to see available algorithms."
        )


@app.command()
def list(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Include models descriptions"
    ),
):
    """
    List all available algorithms.
    """
    typer.echo("Available algorithms:")
    for metadata in ALGORITHMS.values():
        s = f"- {metadata.name}"
        if metadata.description and verbose:
            s += f": {metadata.description}"

        typer.echo(s)

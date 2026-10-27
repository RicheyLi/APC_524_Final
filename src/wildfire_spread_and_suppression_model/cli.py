"""Console script for wildfire_spread_and_suppression_model."""

import typer
from rich.console import Console

from wildfire_spread_and_suppression_model import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for wildfire_spread_and_suppression_model."""
    console.print("Replace this message by putting your code into wildfire_spread_and_suppression_model.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()

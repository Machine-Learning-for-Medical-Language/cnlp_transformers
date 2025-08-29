from typing import Annotated

import rich
import typer
from typer.core import DEFAULT_MARKUP_MODE

from .. import __version__ as cnlpt_version
from . import rest, train

app = typer.Typer(add_completion=False, rich_markup_mode=DEFAULT_MARKUP_MODE)


app.command(no_args_is_help=True)(rest.rest)
app.command(
    no_args_is_help=True,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    epilog=train.TRAIN_EPILOG,
)(train.train)


def version_callback(version: bool):
    if version:
        rich.print(f"cnlp_transformers version: [b cyan]{cnlpt_version}")
        raise typer.Exit()


@app.callback(no_args_is_help=True)
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the cnlp_transformers version and exit.",
            is_eager=True,
            callback=version_callback,
        ),
    ] = False,
):
    pass


def main():
    app()

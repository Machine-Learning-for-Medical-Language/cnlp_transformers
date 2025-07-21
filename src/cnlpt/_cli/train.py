import click

from ..train_system.cnlp_train_system import main as train_system


@click.command(
    "train",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
    add_help_option=False,
)
@click.argument("train_args", nargs=-1, type=click.UNPROCESSED)
def train_command(train_args: list[str]):
    "Fine-tune models for clinical NLP."

    train_system(argv=list(train_args))

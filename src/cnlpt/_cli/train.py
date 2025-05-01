import sys

import click

# from ..train_system import __file__ as train_system_file
# from ..train_system import main as train_system
from ..new_train_system.cnlp_train_system import main as train_system


@click.command(
    "train",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@click.argument("train_args", nargs=-1, type=click.UNPROCESSED)
def train_command(train_args):
    "Fine-tune models for clinical NLP."

    sys.argv = list(train_args)
    train_system()

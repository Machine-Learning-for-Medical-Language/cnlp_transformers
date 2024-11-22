import click

from .. import __version__
from .rest import rest_command


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    type=bool,
    is_flag=True,
    default=False,
    help="Print the cnlp_transformers version.",
)
@click.pass_context
def cli(ctx: click.Context, version: bool):
    if ctx.invoked_subcommand is not None:
        return

    if version:
        print(__version__)
        ctx.exit()
    else:
        click.echo(ctx.get_help())
        ctx.exit()


cli.add_command(rest_command)

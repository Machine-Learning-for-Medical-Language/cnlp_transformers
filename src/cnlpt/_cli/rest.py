import click

from ..api import MODEL_TYPES, get_rest_app


@click.command("rest", context_settings={"show_default": True})
@click.option(
    "--model-type",
    type=click.Choice(MODEL_TYPES),
    required=True,
)
@click.option(
    "-h",
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host address to serve the REST app.",
)
@click.option(
    "-p", "--port", type=int, default=8000, help="Port to serve the REST app."
)
@click.option(
    "--reload",
    type=bool,
    is_flag=True,
    default=False,
    help="Auto-reload the REST app.",
)
def rest_command(model_type: str, host: str, port: int, reload: bool):
    """Start a REST application from a model."""
    import uvicorn

    uvicorn.run(get_rest_app(model_type), host=host, port=port, reload=reload)

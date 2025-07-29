from typing import Union

import click


def parse_models(
    ctx: click.Context,
    param: click.Parameter,
    value: Union[tuple[str, ...], None],
):
    if value is None:
        return None

    models: list[tuple[str, str]] = []
    for item in value:
        if "=" in item:
            prefix, path = item.split("=", 1)
            if not prefix.startswith("/"):
                raise click.BadParameter(
                    f"route prefix must start with '/': {prefix}", param=param
                )
        elif len(value) > 1:
            raise click.BadParameter(
                "route prefixes are required when serving more than one model",
                param=param,
            )
        else:
            path = item
            prefix = ""
        models.append((prefix, path))
    return models


@click.command("rest", context_settings={"show_default": True})
@click.option(
    "--model",
    "models",
    multiple=True,
    callback=parse_models,
    help="Model definition as [ROUTER_PREFIX=]PATH_TO_MODEL. Prefix must start with '/'.",
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
def rest_command(models: list[tuple[str, str]], host: str, port: int):
    """Start a REST application from a model."""
    import uvicorn

    from ..rest import CnlpRestApp

    app = CnlpRestApp.multi_app(
        [(CnlpRestApp(model_path=path), prefix) for prefix, path in models]
    )
    uvicorn.run(app, host=host, port=port)

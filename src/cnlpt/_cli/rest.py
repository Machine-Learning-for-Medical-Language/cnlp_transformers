from typing import Annotated

import typer


def parse_models(ctx: typer.Context, param: typer.CallbackParam, value: str):
    if value is None:
        return None

    models: list[tuple[str, str]] = []
    if isinstance(value, str):
        value = [value]
    for item in value:
        if "=" in item:
            prefix, path = item.split("=", 1)
            if not prefix.startswith("/"):
                raise typer.BadParameter(
                    f"route prefix must start with '/': {prefix}", param=param
                )
        elif len(value) > 1:
            raise typer.BadParameter(
                "route prefixes are required when serving more than one model",
                param=param,
            )
        else:
            path = item
            prefix = ""
        models.append((prefix, path))
    return models


def rest(
    models: Annotated[
        list[str],
        typer.Option(
            "--model",
            callback=parse_models,
            help="Model definition as [ROUTER_PREFIX=]PATH_TO_MODEL. Route prefix must start with '/'. This option can be specified multiple times to serve multiple models simultaneously. Route prefixes are required when serving more than one model.",
        ),
    ],
    host: Annotated[
        str, typer.Option("-h", "--host", help="Host address to serve the REST app.")
    ] = "0.0.0.0",
    port: Annotated[
        int, typer.Option("-p", "--port", help="Port to serve the REST app.")
    ] = 8000,
):
    """Start a REST application from a model."""
    import uvicorn

    from ..rest import CnlpRestApp

    app = CnlpRestApp.multi_app(
        [(CnlpRestApp(model_path=path), prefix) for prefix, path in models]
    )
    uvicorn.run(app, host=host, port=port)

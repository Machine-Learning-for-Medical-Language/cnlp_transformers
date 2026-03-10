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
            help=(
                "Model to serve, specified as PATH_TO_MODEL or ROUTE_PREFIX=PATH_TO_MODEL. "
                "PATH_TO_MODEL can be a local directory or a HuggingFace model repository (e.g. 'mlml-chip/negation_pubmedbert_sharpseed'). "
                "ROUTE_PREFIX must start with '/' and is required when serving more than one model. "
                "This option can be repeated to serve multiple models simultaneously "
                "(e.g. --model /negation=mlml-chip/negation_pubmedbert_sharpseed --model /temporal=mlml-chip/thyme2_colon_e2e)."
            ),
        ),
    ],
    host: Annotated[
        str, typer.Option("-h", "--host", help="Host address to serve the REST app.")
    ] = "localhost",
    port: Annotated[
        int, typer.Option("-p", "--port", help="Port to serve the REST app.")
    ] = 8000,
):
    """Start a REST API server for one or more cnlpt models.

    Serves a FastAPI application with a /process endpoint that accepts text
    (and optionally entity spans) and returns model predictions. Interactive
    API documentation is available at /docs once the server is running.

    \b
    Examples:

      Serve a single model from HuggingFace:

        cnlpt rest --model mlml-chip/negation_pubmedbert_sharpseed

      Serve a single model from a local directory:

        cnlpt rest --model ./my_model --host 0.0.0.0 --port 9000

      Serve multiple models simultaneously, each under its own route prefix:

        cnlpt rest \\
          --model /negation=mlml-chip/negation_pubmedbert_sharpseed \\
          --model /temporal=mlml-chip/thyme2_colon_e2e

    When serving multiple models, each model's /process endpoint is available
    at ROUTE_PREFIX/process (e.g. /negation/process).

    Interactive API documentation for all models is available at HOST:PORT/docs.
    """
    import asyncio
    import logging

    import click
    import uvicorn

    from ..rest import CnlpRestApp

    app = CnlpRestApp.multi_app(
        [(CnlpRestApp(model_path=path), prefix) for prefix, path in models]
    )

    async def serve():
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        task = asyncio.create_task(server.serve())

        # Wait until server is fully started
        while not server.started:
            await asyncio.sleep(0.1)

        logger = logging.getLogger("uvicorn.error")
        docs_addr_format = "http://%s:%s/docs"
        logger.info(
            f"Point your browser at {docs_addr_format} for interactive documentation.",
            host,
            port,
            extra={
                "color_message": f"Point your browser at {click.style(docs_addr_format, bold=True)} for interactive documentation."
            },
        )

        await task

    asyncio.run(serve())

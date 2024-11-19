import click

from ..api.cnn_rest import app as cnn_app
from ..api.current_rest import app as current_app
from ..api.dtr_rest import app as dtr_app
from ..api.event_rest import app as event_app
from ..api.hier_rest import app as hier_app
from ..api.negation_rest import app as negation_app
from ..api.temporal_rest import app as temporal_app
from ..api.termexists_rest import app as termexists_app
from ..api.timex_rest import app as timex_app

APPS = {
    "cnn": cnn_app,
    "current": current_app,
    "dtr": dtr_app,
    "event": event_app,
    "hier": hier_app,
    "negation": negation_app,
    "temporal": temporal_app,
    "termexists": termexists_app,
    "timex": timex_app,
}


@click.command("rest", context_settings={"show_default": True})
@click.option(
    "--model",
    type=click.Choice(APPS.keys()),
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
def rest_command(model: str, host: str, port: int, reload: bool):
    """Start a REST application from a model."""
    import uvicorn

    uvicorn.run(APPS[model], host=host, port=port, reload=reload)

"""Serve REST APIs for CNLPT models over your network."""

from typing import Final

MODEL_TYPES: Final = (
    "cnn",
    "current",
    "dtr",
    # "event",
    "hier",
    "negation",
    "temporal",
    # "termexists",
    # "timex",
)
"""The available model types for :func:`get_rest_app`."""


def get_rest_app(model_type: str):
    """Get a FastAPI app for a certain model type.

    Args:
        model_type: The type of model to serve.

    Returns:
        The FastAPI app.
    """
    if model_type == "cnn":
        from .cnn_rest import app

        return app
    elif model_type == "current":
        from .current_rest import app

        return app
    elif model_type == "dtr":
        from .dtr_rest import app

        return app
    # elif model_type == "event":
    #     from .event_rest import app

    #     return app
    elif model_type == "hier":
        from .hier_rest import app

        return app
    elif model_type == "negation":
        from .negation_rest import app

        return app
    elif model_type == "temporal":
        from .temporal_rest import app

        return app
    # elif model_type == "termexists":
    #     from .termexists_rest import app

    #     return app
    # elif model_type == "timex":
    #     from .timex_rest import app

    #     return app
    else:
        raise ValueError(f"unknown model type: {model_type}")


__all__ = [
    "MODEL_TYPES",
    "get_rest_app",
]

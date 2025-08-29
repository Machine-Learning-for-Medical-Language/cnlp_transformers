import inspect
import logging
import random
import warnings
from typing import Any

from .. import __version__ as cnlpt_version

logger = logging.getLogger(__name__)


def warn_on_version_mismatch(model_version: str):
    ckpt_maj_min = tuple(model_version.split(".", maxsplit=2)[:2])
    cnlpt_maj_min = tuple(cnlpt_version.split(".", maxsplit=2)[:2])

    if ckpt_maj_min != cnlpt_maj_min:
        warning = f"You are loading a model created with cnlpt version {model_version}, but this is version {cnlpt_version}. Be aware that the checkpoint may be incompatible."

        warnings.warn(warning)
        logger.warning(warning)


def generalize_encoder_forward_kwargs(encoder, **kwargs: Any) -> dict[str, Any]:
    """Create a new input feature argument that preserves only the features that are valid for this encoder.
    Warn if a feature is present but not valid for the encoder.

    Args:
        encoder: A HF encoder model

    Returns:
        Dictionary of valid arguments for this encoder
    """
    new_kwargs = dict()
    params = inspect.signature(encoder.forward).parameters
    for name, value in kwargs.items():
        if name not in params and value is not None:
            # Warn if a contentful parameter is not valid
            logger.warning(
                f"Parameter {name} not present for encoder class {encoder.__class__.__name__}."
            )
        elif name in params:
            # Pass all, and only, parameters that are valid,
            # regardless of whether they are None
            new_kwargs[name] = value
        # else, value is None and not in params, so we ignore it
    return new_kwargs


def freeze_encoder_weights(encoder, freeze_prob: float):
    for param in encoder.parameters():
        if random.random() < freeze_prob:
            param.requires_grad = False

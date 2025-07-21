import json
import logging
import os
from typing import Any, Union

import numpy as np
from transformers.models.auto.configuration_auto import AutoConfig

from .. import __version__ as cnlpt_version


def is_cnlpt_model(model_path: str) -> bool:
    """Infer whether a model path refers to a cnlpt
    model checkpoint (if not, we assume it is an
    encoder)

    Args:
        model_path: The path to the model.

    Returns:
        Whether the model is a cnlpt classifier model.
    """
    encoder_config = AutoConfig.from_pretrained(model_path)
    return encoder_config.model_type == "cnlpt"


def is_external_encoder(model_name_or_path: str) -> bool:
    """Check whether a specified model is not a cnlpt model -- an external model like a
    huggingface hub model or a downloaded local directory.

    Args:
        model_name_or_path: Specified model.

    Returns:
        Whether the encoder is an external (non-cnlpt) model.
    """
    return not is_cnlpt_model(model_name_or_path)


def simple_softmax(x: list):
    """Softmax values for 1-D score array"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def warn_if_chekpoint_version_mismatch(
    checkpoint_dir: str, logger: logging.Logger
) -> Union[str, None]:
    warning: Union[str, None] = None

    checkpoint_config_path = os.path.join(checkpoint_dir, "config.json")
    with open(checkpoint_config_path) as f:
        checkpoint_config: dict[str, Any] = json.load(f)

    ckpt_version: Union[str, None] = checkpoint_config.get("cnlpt_version", None)
    if ckpt_version is None:
        warning = f"The checkpoint at {checkpoint_dir} does not specify a `cnlpt_version`, and may be incompatible with this version of cnlpt"
        return
    else:
        ckpt_maj_min = tuple(ckpt_version.split(".", maxsplit=2)[:2])
        cnlpt_maj_min = tuple(cnlpt_version.split(".", maxsplit=2)[:2])

        if ckpt_maj_min != cnlpt_maj_min:
            warning = f"The checkpoint at {checkpoint_dir} was created with cnlpt version {ckpt_version}, but this is version {cnlpt_version}. Be aware that the checkpoint may be incompatible."

    if warning is not None:
        import warnings

        warnings.warn(warning)
        logger.warning(warning)

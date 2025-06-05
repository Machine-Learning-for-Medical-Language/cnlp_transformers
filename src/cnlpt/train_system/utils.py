import numpy as np
from transformers.models.auto.configuration_auto import AutoConfig


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

from transformers import AutoConfig


def is_cnlpt_model(model_path: str) -> bool:
    """
    Infer whether a model path refers to a cnlpt
    model checkpoint (if not, we assume it is an
    encoder)
    :param model_path: the path to the model
    :return: whether the model is a cnlpt classifier model
    """
    encoder_config = AutoConfig.from_pretrained(model_path)
    return encoder_config.model_type == "cnlpt"


def is_external_encoder(model_name_or_path: str) -> bool:
    """
    Check whether a specified model is not a cnlpt model -- an external model like a
    huggingface hub model or a downloaded local directory.
    :param model_name_or_path: specified model
    :return: whether the encoder is an external (non-cnlpt) model
    """
    return not is_cnlpt_model(model_name_or_path)

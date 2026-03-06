from __future__ import annotations

from dataclasses import asdict
from os import PathLike
from typing import Any, Union

from transformers import CONFIG_MAPPING, AutoConfig, AutoModel, PretrainedConfig
from transformers import logging as transformers_logging
from transformers.modeling_utils import PreTrainedModel

from ... import __version__ as current_cnlpt_version
from ...data.task_info import TaskInfo
from ..utils import warn_on_version_mismatch


def _load_encoder_config(
    encoder_config: Union[PretrainedConfig, dict[str, Any], None],
    encoder_name: str,
) -> PretrainedConfig:
    if (
        isinstance(encoder_config, dict)
        and (model_type := encoder_config.get("model_type")) is not None
    ):
        config_class: type[PretrainedConfig] = CONFIG_MAPPING[model_type]
        return config_class.from_dict(encoder_config)
    elif isinstance(encoder_config, PretrainedConfig):
        return encoder_config
    return AutoConfig.from_pretrained(encoder_name)


def _load_tasks(
    tasks: Union[list[dict[str, Any]], list[TaskInfo]],
) -> list[dict[str, Any]]:
    if tasks is None or len(tasks) == 0:
        return []
    elif isinstance(tasks[0], TaskInfo):
        return [asdict(t) for t in tasks]
    else:
        return tasks


class BaseConfig(PretrainedConfig):
    def __init__(
        self,
        *,
        tasks: Union[list[dict[str, Any]], list[TaskInfo], None] = None,
        vocab_size: Union[int, None] = None,
        cnlpt_version: Union[str, None] = None,
        **kwargs,
    ):
        if cnlpt_version is None:
            self.cnlpt_version = current_cnlpt_version
        else:
            warn_on_version_mismatch(cnlpt_version)
            self.cnlpt_version = cnlpt_version

        if "_tasks" in kwargs:
            self._tasks = kwargs.pop("_tasks")
        else:
            self.tasks = tasks

        super().__init__(vocab_size=vocab_size, **kwargs)

    @property
    def tasks(self) -> list[TaskInfo]:
        if self._tasks is None:
            return []
        return [TaskInfo(**t) for t in self._tasks]

    @tasks.setter
    def tasks(self, tasks: Union[list[dict[str, Any]], list[TaskInfo]]):
        if tasks is None or len(tasks) == 0:
            self._tasks = []
        elif isinstance(tasks[0], TaskInfo):
            self._tasks = [asdict(t) for t in tasks]
        else:
            self._tasks = tasks


class BaseConfigWithEncoder(BaseConfig):
    def __init__(
        self,
        *,
        tasks: Union[list[dict[str, Any]], list[TaskInfo], None] = None,
        vocab_size: Union[int, None] = None,
        cnlpt_version: Union[str, None] = None,
        encoder_name: Union[str, PathLike] = "roberta-base",
        encoder_config: Union[PretrainedConfig, dict[str, Any], None] = None,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks,
            vocab_size=vocab_size,
            cnlpt_version=cnlpt_version,
            **kwargs,
        )

        self._set_encoder(encoder_name, encoder_config)

    def _set_encoder(
        self,
        encoder_name: str,
        encoder_config: Union[PretrainedConfig, dict[str, Any], None],
    ):
        self.encoder_name = encoder_name
        self.encoder_config = _load_encoder_config(encoder_config, encoder_name)
        self.encoder_output_dim: int = self._get_encoder_attr("dim", "hidden_size")
        self.encoder_dropout: float = self._get_encoder_attr(
            "dropout", "mlp_dropout", "hidden_dropout_prob"
        )

    def _get_encoder_attr(self, *keys):
        for key in keys:
            if (result := getattr(self.encoder_config, key, None)) is not None:
                return result
        raise ValueError(
            f"Encoder config does not have any of the attributes {[*keys]}. "
            "Please use a supported encoder (e.g. BERT/RoBERTa/DistilBERT/ModernBERT)"
        )

    def load_encoder_model(self, resize_token_embeddings: bool) -> PreTrainedModel:
        # Disable warnings for a moment while we load the model to keep the console clean.
        # (The emitted warnings are non-issues.)
        verb_before = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        encoder: PreTrainedModel = AutoModel.from_config(config=self.encoder_config)
        if resize_token_embeddings:
            encoder.resize_token_embeddings(self.vocab_size, mean_resizing=False)
        transformers_logging.set_verbosity(verb_before)
        return encoder

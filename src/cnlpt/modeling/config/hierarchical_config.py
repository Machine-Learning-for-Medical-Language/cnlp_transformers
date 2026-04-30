import os
from typing import Any

from ...data.task_info import TaskInfo
from .base_config import BaseConfigWithEncoder


def _resolve_layer(layer: int, n_layers: int):
    if layer < 0:
        layer = layer + n_layers + 1

    if layer > n_layers:
        raise ValueError(
            f"The layer specified ({layer}) is too big for the specified chunk transformer which has {n_layers} layers"
        )
    elif layer < 0:
        raise ValueError(
            f"The layer specified ({layer}) is a negative value which is larger than the actual number of layers {n_layers}"
        )
    elif layer == 0:
        raise ValueError(
            "The classifier layer derived is 0 which is ambiguous -- there is no usable 0th layer in a hierarchical model. Enter a value for the layer argument that at least 1 (use one layer) or -1 (use the final layer)"
        )
    return layer


class HierarchicalModelConfig(BaseConfigWithEncoder):
    model_type = "cnlpt.hier"

    def __init__(
        self,
        *,
        tasks: list[dict[str, Any]] | list[TaskInfo] | None = None,
        vocab_size: int | None = None,
        encoder_name: str | os.PathLike = "roberta-base",
        layer: int = -1,
        n_layers: int = 8,
        d_inner: int = 2048,
        n_head: int = 8,
        d_k: int = 8,
        d_v: int = 96,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks,
            vocab_size=vocab_size,
            encoder_name=encoder_name,
            **kwargs,
        )

        self.layer = _resolve_layer(layer, n_layers)
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout

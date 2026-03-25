from typing import Any

from ...data.task_info import CLASSIFICATION, TaskInfo
from .base_config import BaseConfig


class LstmModelConfig(BaseConfig):
    model_type = "cnlpt.lstm"

    def __init__(
        self,
        *,
        tasks: list[dict[str, Any]] | list[TaskInfo] | None = None,
        vocab_size: int | None = None,
        embed_dim: int = 100,
        hidden_size: int = 100,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks,
            vocab_size=vocab_size,
            **kwargs,
        )

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout

        if any(t.type != CLASSIFICATION for t in self.tasks):
            raise NotImplementedError(
                "using a LSTM model for non-classification tasks is not yet implemented"
            )

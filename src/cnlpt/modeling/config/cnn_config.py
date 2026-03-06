from typing import Any, Union

from ...data.task_info import CLASSIFICATION, TaskInfo
from .base_config import BaseConfig


class CnnModelConfig(BaseConfig):
    model_type = "cnlpt.cnn"

    def __init__(
        self,
        *,
        tasks: Union[list[dict[str, Any]], list[TaskInfo], None] = None,
        vocab_size: Union[int, None] = None,
        use_prior_tasks: bool = False,
        embed_dim: int = 100,
        num_filters: int = 25,
        filter_sizes: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks,
            vocab_size=vocab_size,
            **kwargs,
        )

        self.use_prior_tasks = use_prior_tasks
        self.embed_dim = embed_dim
        self.filters_per_size = num_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout

        if any(t.type != CLASSIFICATION for t in self.tasks):
            raise NotImplementedError(
                "using a CNN model for non-classification tasks is not yet implemented"
            )

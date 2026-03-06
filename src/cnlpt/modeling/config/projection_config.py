import os
from typing import Any, Literal, Union

from ...data.task_info import TaskInfo
from .base_config import BaseConfigWithEncoder


class ProjectionModelConfig(BaseConfigWithEncoder):
    model_type = "cnlpt.proj"

    def __init__(
        self,
        *,
        tasks: Union[list[dict[str, Any]], list[TaskInfo], None] = None,
        vocab_size: Union[int, None] = None,
        encoder_name: Union[str, os.PathLike] = "roberta-base",
        encoder_layer: int = -1,
        use_prior_tasks: bool = False,
        classification_mode: Literal["cls", "tagged"] = "cls",
        num_rel_attention_heads: int = 12,
        rel_attention_head_dims: int = 64,
        character_level: bool = False,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks,
            vocab_size=vocab_size,
            encoder_name=encoder_name,
            **kwargs,
        )

        self.encoder_layer = encoder_layer
        self.use_prior_tasks = use_prior_tasks
        self.tokens = (
            classification_mode == "tagged"
        )  # TODO(ian) this should really be self.classification_mode
        self.num_rel_attention_heads = num_rel_attention_heads
        self.rel_attention_head_dims = rel_attention_head_dims
        self.character_level = character_level

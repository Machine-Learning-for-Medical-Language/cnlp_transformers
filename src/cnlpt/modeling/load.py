from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from ..data.task_info import CLASSIFICATION, RELATIONS, TAGGING, TaskInfo
from .config import (
    CnnModelConfig,
    HierarchicalModelConfig,
    LstmModelConfig,
    ProjectionModelConfig,
)


def try_load_config(
    model_name_or_path: str,
) -> ProjectionModelConfig | CnnModelConfig | HierarchicalModelConfig | LstmModelConfig:
    """Load a model config, potentially for a model created with an earlier version of CNLPT.

    Args:
        config_file: Path to a config file on disk.

    Returns:
        The loaded config.
    """

    config_data = PretrainedConfig.get_config_dict(model_name_or_path)[0]

    if "model_type" not in config_data:
        raise ValueError("could not infer model type")
    model_type: str = config_data.pop("model_type")

    if model_type.startswith("cnlpt."):
        # This is a post-0.7 model, so we'll just use autoconfig
        return AutoConfig.from_pretrained(model_name_or_path)

    config_data.pop("architectures")

    training_tasks: list[str] = config_data.pop("finetuning_task")
    label_dict: dict[str, list[str]] = config_data.pop("label_dictionary")
    tagging_tasks: dict[str, bool] = config_data.pop("tagger")
    relations_tasks: dict[str, bool] = config_data.pop("relations")

    tasks: list[TaskInfo] = []
    for task_idx, task_name in enumerate(training_tasks):
        task_type = (
            TAGGING
            if tagging_tasks[task_name]
            else RELATIONS
            if relations_tasks[task_name]
            else CLASSIFICATION
        )
        tasks.append(
            TaskInfo(
                name=task_name,
                type=task_type,
                index=task_idx,
                labels=tuple(label_dict[task_name]),
            )
        )

    config_data["tasks"] = tasks

    if model_type == "cnlpt":
        config_data["architectures"] = ["ProjectionModel"]
        config_data["model_type"] = "cnlpt.proj"
        tagged_mode = config_data.pop("tokens")
        config_data["classification_mode"] = "tagged" if tagged_mode else "cls"
        config_data["encoder_layer"] = config_data.pop("layer")
        return ProjectionModelConfig(**config_data)
    else:
        raise NotImplementedError(
            "loading legacy models other than projection models is not yet implemented"
        )


def try_load_pretrained_model(pretrained_model_name_or_path: str) -> PreTrainedModel:
    config = try_load_config(pretrained_model_name_or_path)
    return AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)

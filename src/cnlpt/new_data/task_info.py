from dataclasses import dataclass
from functools import cached_property
from typing import Final, Literal, Union

TaskType = Literal["classification", "tagging", "relations"]

CLASSIFICATION: Final[TaskType] = "classification"
TAGGING: Final[TaskType] = "tagging"
RELATIONS: Final[TaskType] = "relations"


def get_task_type(task_type_str: str) -> TaskType:
    task_type_str = task_type_str.lower()

    if task_type_str == "classification":
        return CLASSIFICATION
    elif task_type_str == "tagging":
        return TAGGING
    elif task_type_str in ("relations", "relex"):
        return RELATIONS
    raise ValueError(f"unknown task type: {task_type_str}")


@dataclass(frozen=True)
class TaskInfo:
    name: str
    type: TaskType
    index: int
    labels: tuple[str, ...]

    @cached_property
    def _label_to_id(self):
        return {label: i for i, label in enumerate(self.labels)}

    def get_label_id(self, label: str, specials: Union[dict[str, int], None] = None):
        if specials and label in specials:
            return specials[label]
        return self._label_to_id[label]

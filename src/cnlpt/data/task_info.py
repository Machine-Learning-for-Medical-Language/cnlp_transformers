from dataclasses import dataclass
from functools import cached_property
from typing import Final, Literal, Union

TaskType = Literal["classification", "tagging", "relations"]
"A type of task that this library can train a model to solve."

CLASSIFICATION: Final[TaskType] = "classification"
"TaskType for sequence classification."

TAGGING: Final[TaskType] = "tagging"
"TaskType for sequence tagging."

RELATIONS: Final[TaskType] = "relations"
"TaskType for relation extraction."


def get_task_type(task_type_str: str) -> TaskType:
    """Convert a string to a `TaskType`.

    Normalizes case and handles interpreting "relex" as "relations".
    """
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
    """Information describing a cnlp training task.

    Attributes:
        name: The name of the task.
        type: The task's `TaskType`, e.g., "classification".
        index: This task's index among all the currently active tasks.
        labels: A sorted tuple of unique label values for this task.
    """

    name: str
    type: TaskType
    index: int
    labels: tuple[str, ...]

    @cached_property
    def _label_to_id(self):
        return {label: i for i, label in enumerate(self.labels)}

    def get_label_id(self, label: str, specials: Union[dict[str, int], None] = None):
        """Get a unique integer id for one of this task's labels.

        Args:
            label: One of this task's labels, or a special label that is one of the keys in `specials`.
            specials: An optional mapping from special labels to their ids. This special id mapping will take precedence over the default id mapping.
        """

        if specials and label in specials:
            return specials[label]
        return self._label_to_id[label]

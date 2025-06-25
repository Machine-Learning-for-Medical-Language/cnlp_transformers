import json
import os
from collections.abc import Iterable
from typing import Any, Final, Literal, Union, cast

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .log import logger
from .task_info import (
    CLASSIFICATION,
    RELATIONS,
    TAGGING,
    TaskInfo,
    TaskType,
    get_task_type,
)

DatasetSplit = Literal["train", "test", "validation"]
CNLP_FILE_FORMATS: Final = ("json", "csv", "tsv")
RESERVED_COLUMN_NAMES: Final = ("id", "text", "text_a", "text_b")
NONE_VALUE: Final = "__None__"


def _infer_split(filepath: Union[str, os.PathLike]) -> DatasetSplit:
    _dir, filename = os.path.split(filepath)
    root, _ext = os.path.splitext(filename)

    if root in ("train", "test", "validation"):
        return root

    # other accepted filenames for validation data
    if root in ("valid", "dev"):
        return "validation"

    raise ValueError(f"unable to infer split (train/test/validation) from {filepath}")


def _infer_task_type_from_labels(labels: set[str]) -> TaskType:
    if any(label.endswith(")") for label in labels):
        return RELATIONS
    elif any(" " in label for label in labels):
        return TAGGING
    else:
        return CLASSIFICATION


def _infer_tasks(dataset: Dataset) -> list[TaskInfo]:
    tasks: list[TaskInfo] = []

    column_names = [c for c in dataset.column_names if c not in RESERVED_COLUMN_NAMES]
    for i, column_name in enumerate(column_names):
        raw_labels = set(str(label) for label in dataset[column_name])
        task_type = _infer_task_type_from_labels(raw_labels)
        labels = _get_sorted_label_set(raw_labels, task_type)

        tasks.append(
            TaskInfo(name=column_name, type=task_type, index=i, labels=tuple(labels))
        )

    return tasks


def _get_sorted_label_set(all_raw_labels: Iterable[str], task_type: TaskType):
    if task_type == CLASSIFICATION:
        label_set = set(all_raw_labels)
    elif task_type == TAGGING:
        joined = " ".join(all_raw_labels)
        label_set = set(joined.split(" "))
    elif task_type == RELATIONS:
        joined = " , ".join(all_raw_labels)
        label_set = set(
            rel_or_none.removesuffix(")").split(",")[-1]
            for rel_or_none in joined.split(" , ")
        )
        label_set.add("None")
    else:
        raise ValueError(f"invalid task type {TaskType}")

    return sorted(label_set)


class CnlpDataReader:
    """Utility class to read and parse raw CNLP-formatted data.

    This class will read CNLP-formatted data from disk and collect it into a `DatasetDict`,
    as well as maintain a list of `TaskInfo` instances representing the data's CNLP tasks.
    """

    def __init__(self, allow_disjoint_labels: bool = False):
        """Initialize an empty `CnlpDataReader` with no data."""
        self.dataset = DatasetDict()
        self.allow_disjoint_labels = allow_disjoint_labels
        self._tasks: list[TaskInfo] = []

    @property
    def split_names(self) -> set[DatasetSplit]:
        """The set of splits (i.e., "train", "validation", and/or "test") present in the data."""
        return set(self.dataset.keys())

    @property
    def task_names(self):
        """The names of all the tasks present in the data."""
        return tuple(t.name for t in self._tasks)

    def _get_task_by_name(self, task_name: str):
        for task in self._tasks:
            if task.name == task_name:
                return task
        raise ValueError(f'task with name "{task_name}" not found')

    def get_tasks(self, task_names: Union[Iterable[str], None] = None):
        """Get all or some subset of the tasks in the data.

        The `TaskInfo` objects returned by this method will have their `index` property
        set to their index in the returned tuple.

        Args:
            task_names: The names of the tasks to retrieve. If set to None, retrieves all tasks.
                Defaults to None.

        Returns:
            A tuple of `TaskInfo` objects.
        """
        if task_names is None:
            return tuple(self._tasks)
        result: list[TaskInfo] = []
        for i, task_name in enumerate(task_names):
            t = self._get_task_by_name(task_name)
            result.append(TaskInfo(name=t.name, type=t.type, index=i, labels=t.labels))
        return tuple(result)

    def _extend(self, new_dataset: DatasetDict, tasks: list[TaskInfo]):
        # first merge the tasks
        for new_task in tasks:
            if new_task.name not in self.task_names:
                self._tasks.append(
                    TaskInfo(
                        name=new_task.name,
                        type=new_task.type,
                        index=len(self._tasks),
                        labels=new_task.labels,
                    )
                )
            else:
                existing = next(t for t in self._tasks if t.name == new_task.name)

                if new_task.type != existing.type:
                    raise ValueError(
                        f'the task "{existing.name}" has two different output modes in different datasets '
                        + f'and might not be the same task: "{existing.name}" ({existing.type}) vs. "{new_task.name}" ({new_task.type})'
                    )
                existing_label_set = set(existing.labels)
                new_label_set = set(new_task.labels)
                if existing_label_set == new_label_set:
                    # the name, output type, and labels are all the same,
                    # so we don't have to do anything.
                    continue
                elif (
                    existing_label_set.issubset(new_label_set)
                    or new_label_set.issubset(existing_label_set)
                    or self.allow_disjoint_labels
                ):
                    logger.warning(
                        f'two different datasets have the same task name "{existing.name}" but not completely equal label lists: '
                        + f"{sorted(existing_label_set)!s} vs. {sorted(new_label_set)!s}. We will merge them."
                    )
                    self._tasks[existing.index] = TaskInfo(
                        name=existing.name,
                        type=existing.type,
                        index=existing.index,
                        labels=tuple(sorted(existing_label_set.union(new_label_set))),
                    )
                else:
                    raise ValueError(
                        f"the task {existing.name} has disjoint sets of labels in different datasets: "
                        + f"{sorted(existing_label_set)!s} vs. {sorted(new_label_set)!s}"
                    )

        # ensure all splits have all columns
        to_merge = [self.dataset, new_dataset]
        for dataset in to_merge:
            for split in dataset.keys():
                for task in self._tasks:
                    if task.name not in dataset[split].column_names:
                        dataset[split] = dataset[split].add_column(
                            task.name,
                            [NONE_VALUE] * len(dataset[split]),
                        )  # type: ignore (suppress type warning about `new_fingerprint` argument handled by internal decorator)

        splits: set[DatasetSplit] = self.split_names.union(new_dataset.keys())
        self.dataset = DatasetDict(
            **{
                split: concatenate_datasets([d[split] for d in to_merge if split in d])
                for split in splits
            }
        )

    def load_json(
        self,
        json_filepath: Union[str, os.PathLike],
        split: Union[DatasetSplit, None] = None,
    ):
        """Update this reader with new data from a CNLP-formatted json file.

        Args:
            json_filepath: The path to the json file.
            split: Which split this data should be a part of. If None, the split will be inferred by the file name. Defaults to None.

        Raises:
            ValueError: If the file is improperly formatted.
        """
        if split is None:
            split = _infer_split(json_filepath)

        dataset = cast(
            DatasetDict,
            load_dataset(
                path="json",
                data_files={split: os.fspath(json_filepath)},
                field="data",
            ),
        )

        tasks: list[TaskInfo] = []

        # For json files, we'll try to get task metadata from the file, then from an adjacent metadata.json.
        # If no metadata is found in either of those locations, we'll fall back to inferring it like we do
        # for csv/tsv datasets.

        with open(json_filepath) as f:
            data: dict[str, Any] = json.load(f)

        metadata: dict[str, Any] | None = None
        if "metadata" in data:
            metadata = data["metadata"]
        else:
            # no metadata in provided json file, look for an adjacent metadata.json instead
            parent_dir, _ = os.path.split(json_filepath)
            metadata_filepath = os.path.join(parent_dir, "metadata.json")
            if os.path.exists(metadata_filepath):
                with open(metadata_filepath) as f:
                    metadata = json.load(f)

        if metadata:
            if "subtasks" not in metadata:
                raise ValueError(
                    f'"subtasks" field is missing from metadata for {json_filepath}'
                )

            for i, task in enumerate(metadata["subtasks"]):
                task_name = task["task_name"]
                if task_name not in dataset[split].column_names:
                    raise ValueError(
                        f'task "{task}" found in metadata but not in dataset for {json_filepath}'
                    )
                task_type = get_task_type(task["output_mode"])
                label_set = _get_sorted_label_set(dataset[split][task_name], task_type)
                tasks.append(
                    TaskInfo(
                        name=task_name,
                        type=task_type,
                        index=i,
                        labels=tuple(label_set),
                    )
                )
        else:
            logger.warning(
                f"no task metadata found in {json_filepath}, and no metadata.json file was found either -- tasks will be inferred instead"
            )
            tasks = _infer_tasks(dataset[split])

        self._extend(dataset, tasks)

    def load_csv(
        self,
        csv_filepath: Union[str, os.PathLike],
        split: Union[DatasetSplit, None] = None,
        sep: str = ",",
    ):
        """Update this reader with new data from a CNLP-formatted csv (or tsv) file.

        Args:
            csv_filepath: The path to the csv (or tsv) file.
            split: Which split this data should be a part of. If None, the split will be inferred by the file name. Defaults to None.
            sep: The separator to use for reading this file. Defaults to ",". For tsv, this should be set to "\\t".
        """
        if split is None:
            split = _infer_split(csv_filepath)

        dataset = cast(
            DatasetDict,
            load_dataset(
                path="csv",
                sep=sep,
                data_files={split: os.fspath(csv_filepath)},
            ),
        )
        tasks = _infer_tasks(dataset[split])
        self._extend(dataset, tasks)

    def load_dir(self, data_dir: Union[str, os.PathLike]):
        """Update this reader with new data from a directory containing CNLP-formatted data.

        This will search (non-recursively) for files named "train", "test", "validation", "valid", or "dev",
        that have the extension ".csv", ".tsv", or ".json".
        The split and data format will be inferred from the filename for each file.

        Args:
            data_dir: Directory with CNLP-formatted data to load.
        """
        for filename in os.listdir(data_dir):
            root, ext = os.path.splitext(filename)
            ext = ext.removeprefix(".")
            if (
                root in ("train", "test", "validation", "valid", "dev")
                and ext in CNLP_FILE_FORMATS
            ):
                filepath = os.path.join(data_dir, filename)
                if ext == "json":
                    self.load_json(json_filepath=filepath)
                elif ext == "csv":
                    self.load_csv(csv_filepath=filepath, sep=",")
                elif ext == "tsv":
                    self.load_csv(csv_filepath=filepath, sep="\t")

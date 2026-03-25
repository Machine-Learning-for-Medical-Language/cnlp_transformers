import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from datasets import Dataset
from scipy.special import softmax
from transformers.trainer_utils import PredictionOutput

from ..data.preprocess import MASK_VALUE
from .task_info import CLASSIFICATION, TAGGING, TaskInfo


@dataclass
class TaskPredictions:
    task: TaskInfo
    logits: npt.NDArray
    labels: npt.NDArray | None

    @property
    def probs(self) -> npt.NDArray:
        return softmax(self.logits, axis=-1)

    @property
    def predicted_int_labels(self) -> npt.NDArray:
        return np.argmax(self.logits, axis=-1)

    @property
    def predicted_str_labels(self) -> npt.NDArray:
        return np.array(self.task.labels)[self.predicted_int_labels]

    @property
    def target_str_labels(self) -> npt.NDArray | None:
        if self.labels is None:
            return None
        masked = self.labels.copy()
        masked[masked == MASK_VALUE] = len(self.task.labels)
        return np.array((*self.task.labels, "[MASK]"))[masked]


@dataclass
class CnlpPredictions:
    input_data: Dataset
    raw: PredictionOutput
    tasks: list[TaskInfo]

    task_predictions: dict[str, TaskPredictions]

    def __init__(
        self,
        input_data: Dataset,
        raw_prediction: PredictionOutput,
        tasks: Iterable[TaskInfo],
        max_seq_length: int,
    ):
        self.input_data = input_data
        self.raw = raw_prediction
        self.tasks = sorted(tasks, key=lambda t: t.index)
        self.max_seq_length = max_seq_length

        # task indices must start at zero and increase by 1
        if not all(idx == t.index for idx, t in enumerate(tasks)):
            raise RuntimeError("task indices should start at zero and increase by one")

        self.task_predictions: dict[str, TaskPredictions] = {}

        task_labels: dict[str, npt.NDArray | None]

        if self.raw.label_ids is None:
            task_labels = {t.name: None for t in tasks}
        elif self.raw.label_ids.ndim == 2:
            # If our labels are 2 dimensional, then label_ids has shape (batch, n_tasks)
            # and all the tasks must be classification. This reflects how we structure
            # the data during preprocessing.
            assert all(t.type == CLASSIFICATION for t in tasks)
            task_labels = {
                t.name: self.raw.label_ids[:, t.index].astype(int) for t in tasks
            }
        else:
            assert self.raw.label_ids.ndim == 3
            # If our labels are 3 dimensional, then label_ids has shape (batch, max_seq, L)
            # where L = n_classification_tasks + n_tagging_tasks + (max_seq * n_relations_tasks).
            task_labels = {}
            offset = 0
            for task in tasks:
                if task.type == CLASSIFICATION:
                    # for classification tasks we only use the first token in the sequence
                    task_labels[task.name] = self.raw.label_ids[:, 0, offset].astype(
                        int
                    )
                    offset += 1
                elif task.type == TAGGING:
                    task_labels[task.name] = self.raw.label_ids[:, :, offset].astype(
                        int
                    )
                    offset += 1
                else:  # task.type == RELATIONS
                    task_labels[task.name] = self.raw.label_ids[
                        :, :, offset : offset + self.max_seq_length
                    ].astype(int)
                    offset += self.max_seq_length

        self.task_predictions = {
            t.name: TaskPredictions(
                task=t,
                logits=self.raw.predictions[t.index],
                labels=task_labels[t.name].squeeze()
                if task_labels[t.name] is not None
                else None,
            )
            for t in tasks
        }

    def to_dict(self):
        def arr_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [arr_to_list(item) for item in obj]
            else:
                return obj

        return {
            "input_data": self.input_data.to_dict(),
            "raw": {
                "predictions": arr_to_list(self.raw.predictions),
                "label_ids": arr_to_list(self.raw.label_ids),
                "metrics": self.raw.metrics,
            },
            "tasks": [asdict(t) for t in self.tasks],
            "max_seq_length": self.max_seq_length,
        }

    def save_json(
        self,
        json_filepath: str | os.PathLike,
        allow_overwrite: bool = False,
    ):
        write_mode = "w" if allow_overwrite else "x"

        with open(json_filepath, write_mode) as preds_file:
            json.dump(self.to_dict(), preds_file, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        def list_to_arr(obj, dtype):
            if obj is None:
                return None
            return np.array(obj, dtype=dtype)

        input_data = Dataset.from_dict(data["input_data"])
        raw = PredictionOutput(
            predictions=data["raw"]["predictions"],
            label_ids=list_to_arr(data["raw"]["label_ids"], np.int64),
            metrics=data["raw"]["metrics"],
        )
        tasks = [TaskInfo(**t) for t in data["tasks"]]
        max_seq_length = data["max_seq_length"]

        return cls(
            input_data=input_data,
            raw_prediction=raw,
            tasks=tasks,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def load_json(cls, filepath: str | os.PathLike):
        with open(filepath) as f:
            return cls.from_dict(json.load(f))

    @property
    def metrics(self):
        return {k.removeprefix("test_"): v for k, v in self.raw.metrics.items()}

    def metrics_df(self):
        metrics = self.metrics.items()
        return pl.DataFrame(
            {
                "metric": [m for m, v in metrics],
                "value": [v for m, v in metrics],
            }
        )

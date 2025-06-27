import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from datasets import Dataset
from scipy.special import softmax
from transformers.trainer_utils import PredictionOutput

from ..args.data_args import CnlpDataArguments
from .task_info import CLASSIFICATION, TAGGING, TaskInfo


@dataclass
class TaskPrediction:
    task: TaskInfo
    logits: npt.NDArray
    labels: Union[npt.NDArray, None]

    @property
    def probs(self) -> npt.NDArray:
        return softmax(self.logits, axis=-1)

    @property
    def predicted_labels(self) -> npt.NDArray:
        return np.argmax(self.logits, axis=-1)


@dataclass
class CnlpPredictions:
    raw: PredictionOutput
    task_predictions: dict[str, TaskPrediction]

    def __init__(
        self,
        input_data: Dataset,
        tokens: list[list[str]],
        raw_prediction: PredictionOutput,
        tasks: Iterable[TaskInfo],
        data_args: CnlpDataArguments,
    ):
        self.input_data = input_data
        self.tokens = tokens
        self.data_args = data_args

        tasks = sorted(tasks, key=lambda t: t.index)
        # task indices must start at zero and increase by 1
        assert all(idx == t.index for idx, t in enumerate(tasks))

        self.raw = raw_prediction
        self.task_predictions: dict[str, TaskPrediction] = {}

        task_labels: dict[str, npt.NDArray]

        if self.raw.label_ids is None:
            task_labels = {t.name: None for t in tasks}
        elif self.raw.label_ids.ndim == 2:
            # If our labels are 2 dimensional, then label_ids has shape (batch, n_tasks)
            # and all the tasks must be classification. This reflects how we structure
            # the data during preprocessing.
            assert all(t.type == CLASSIFICATION for t in tasks)
            task_labels = {t.name: self.raw.label_ids[:, t.index] for t in tasks}
        else:
            assert self.raw.label_ids.ndim == 3
            # If our labels are 3 dimensional, then label_ids has shape (batch, max_seq, L)
            # where L = n_classification_tasks + n_tagging_tasks + (max_seq * n_relations_tasks).
            task_labels = {}
            offset = 0
            for task in tasks:
                if task.type == CLASSIFICATION:
                    # for classification tasks we only use the first token in the sequence
                    task_labels[task.name] = self.raw.label_ids[:, 0, offset]
                    offset += 1
                elif task.type == TAGGING:
                    task_labels[task.name] = self.raw.label_ids[:, :, offset]
                    offset += 1
                else:  # task.type == RELATIONS
                    task_labels[task.name] = self.raw.label_ids[
                        :, :, offset : offset + self.data_args.max_seq_length
                    ]
                    offset += self.data_args.max_seq_length

        self.task_predictions = {
            t.name: TaskPrediction(
                task=t,
                logits=self.raw.predictions[t.index],
                labels=task_labels[t.name].squeeze(),
            )
            for t in tasks
        }

    def get(self, task_name: str):
        return self.task_predictions[task_name]

    def to_data_frame(
        self,
        *tasks: str,
        include_logits: bool = True,
        include_probs: bool = False,
    ):
        if len(tasks) == 0:
            tasks = self.task_predictions.keys()

        cols: list[pl.Series] = []

        idxs = pl.Series("sample_idx", list(range(len(self.input_data))))
        cols.append(idxs)

        if "id" in self.input_data.column_names:
            cols.append(pl.Series("sample_id", self.input_data["id"]))

        cols.append(
            pl.Series("text", self.input_data["text"]),
        )

        cols.append(
            pl.Series(
                "tokens",
                self.tokens,
                dtype=pl.Array(pl.String, shape=len(self.tokens[0])),
            )
        )

        result = pl.DataFrame(cols)

        for task_name in tasks:
            task_pred = self.get(task_name)
            task_cols = [
                idxs,
                pl.Series("raw_label", self.input_data[task_name]),
                pl.Series("label", task_pred.labels),
                pl.Series("predicted_label", task_pred.predicted_labels),
            ]
            if include_logits:
                task_cols.append(pl.Series("logits", task_pred.logits))
            if include_probs:
                task_cols.append(pl.Series("probs", task_pred.probs))

            task_df = pl.DataFrame(task_cols).select(
                "sample_idx",
                pl.struct(pl.all().exclude("sample_idx")).alias(task_name),
            )
            result = result.join(task_df, on="sample_idx")

        return result

    def save(
        self,
        save_dir: Union[str, os.PathLike],
        save_fmt: Literal["parquet", "csv", "json"] = "parquet",
        include_logits: bool = True,
        include_probs: bool = False,
        allow_overwrite: bool = False,
        polars_write_kwargs: Union[dict[str, Any], None] = None,
    ):
        if not os.path.isdir(save_dir):
            raise ValueError(f"{save_dir:!s} does not exist or is not a directory")

        df = self.to_data_frame(
            include_logits=include_logits, include_probs=include_probs
        )

        write_mode = "w" if allow_overwrite else "x"
        if polars_write_kwargs is None:
            polars_write_kwargs = dict()

        preds_filepath = os.path.join(save_dir, f"predictions.{save_fmt}")
        with open(preds_filepath, write_mode) as preds_file:
            if save_fmt == "parquet":
                df.write_parquet(preds_file, **polars_write_kwargs)
            elif save_fmt == "csv":
                df.write_csv(preds_file, **polars_write_kwargs)
            elif save_fmt == "json":
                df.write_json(preds_file, **polars_write_kwargs)

        if self.raw.metrics is not None:
            metrics_filepath = os.path.join(save_dir, "metrics.json")
            with open(metrics_filepath, write_mode) as metrics_file:
                json.dump(self.raw.metrics, metrics_file, indent=2)

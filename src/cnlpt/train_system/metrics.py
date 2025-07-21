from dataclasses import dataclass
from typing import Union

import numpy as np
from sklearn.metrics import classification_report

from ..data.preprocess import MASK_VALUE
from ..data.task_info import TaskInfo


@dataclass(frozen=True)
class TaskEvalPrediction:
    task: TaskInfo
    predictions: np.ndarray
    probs: Union[np.ndarray, None]
    labels: Union[np.ndarray, None]

    def compute_metrics(self) -> dict[str, float]:
        if self.labels is None:
            raise RuntimeError(
                "cannot compute metrics because eval prediction has no labels"
            )

        if len(self.predictions) != len(self.labels):
            raise RuntimeError(
                f"Predictions and labels have mismatched lengths {len(self.predictions)} and {len(self.labels)}"
            )
        preds = self.predictions.flatten()
        labels = self.labels.flatten().astype("int")

        pred_inds = np.where(labels != MASK_VALUE)
        preds = preds[pred_inds]
        labels = labels[pred_inds]

        report = classification_report(
            y_true=labels,
            y_pred=preds,
            target_names=list(self.task.labels),
            labels=range(len(self.task.labels)),
            output_dict=True,
            zero_division=0,
        )

        task_metrics = {
            "acc": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "micro_f1": report["weighted avg"]["f1-score"],
            **{f"{label}.f1": report[label]["f1-score"] for label in self.task.labels},
        }

        return {f"{self.task.name}.{key}": val for key, val in task_metrics.items()}

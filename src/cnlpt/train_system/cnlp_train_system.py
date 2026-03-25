import contextlib
import math
import os

import numpy as np
import numpy.typing as npt
from datasets import Dataset
from transformers.trainer import Trainer
from transformers.trainer_callback import PrinterCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction, IntervalStrategy

from ..data import RELATIONS, TAGGING, CnlpDataset, CnlpPredictions
from ..modeling.models import CnnModel, HierarchicalModel, LstmModel, ProjectionModel
from .args import CnlpTrainingArguments
from .display import TrainSystemDisplay
from .log import configure_logger_for_training
from .metrics import TaskEvalPrediction
from .training_callbacks import BasicLoggingCallback, DisplayCallback
from .utils import simple_softmax


class CnlpTrainSystem:
    def __init__(
        self,
        model: CnnModel | LstmModel | HierarchicalModel | ProjectionModel,
        dataset: CnlpDataset,
        training_args: CnlpTrainingArguments,
    ):
        self.model = model
        self.dataset = dataset
        self.args = training_args

        self._ensure_model_dataset_compatibility()
        self._set_eval_strategy()

        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)

    def _ensure_model_dataset_compatibility(self):
        if (
            isinstance(self.model, HierarchicalModel)
            and self.dataset.hier_config is None
        ):
            raise ValueError(
                "to train a hierarchical model, you need a hierarchical-formatted dataset. Pass a HierarchicalDataConfig instance to your dataset initializer."
            )
        elif (
            not isinstance(self.model, HierarchicalModel)
            and self.dataset.hier_config is not None
        ):
            raise ValueError(
                "cannot train a non-hierarchical model on a hierarchical-formatted dataset. Be sure not to pass a HierarchicalDataConfig instance to your dataset initializer."
            )

    def _set_eval_strategy(self):
        if self.args.do_train:
            if self.args.evals_per_epoch > 0:
                batches_per_epoch = math.ceil(
                    len(self.dataset.train_data) / self.args.train_batch_size
                )
                total_steps = int(
                    self.args.num_train_epochs
                    * batches_per_epoch
                    // self.args.gradient_accumulation_steps
                )
                steps_per_epoch = int(total_steps // self.args.num_train_epochs)
                self.args.eval_steps = steps_per_epoch // self.args.evals_per_epoch
                self.args.eval_strategy = IntervalStrategy.STEPS
            elif self.args.do_eval:
                self.args.eval_strategy = IntervalStrategy.EPOCH

    def _extract_task_predictions(self, p: EvalPrediction):
        task_predictions: list[TaskEvalPrediction] = []
        task_label_offset = 0

        for task in self.dataset.tasks:
            probs: npt.NDArray[np.float64] | None = None

            raw_preds = p.predictions[task.index]
            if task.type == TAGGING:
                preds = np.argmax(raw_preds, axis=2)
                # labels will be -100 where we don't need to tag
            elif task.type == RELATIONS:
                preds = np.argmax(raw_preds, axis=3)
            else:
                preds = np.argmax(raw_preds, axis=1)
                if self.args.report_probs:
                    probs = np.max(
                        [simple_softmax(logits) for logits in raw_preds],
                        axis=1,
                    )

            labels: npt.NDArray[np.int64] | None
            task_label_width = 0

            label_ids: npt.NDArray[np.int64] | None = getattr(p, "label_ids", None)
            if label_ids is None:
                # we are doing inference, so no labels
                labels = None
            elif task.type == RELATIONS:
                task_label_width = self.dataset.max_seq_length
                # relation labels
                labels = label_ids[
                    :, :, task_label_offset : task_label_offset + task_label_width
                ]
            elif label_ids.ndim == 3:
                task_label_width = 1
                if task.type == TAGGING:
                    labels = label_ids[
                        :, :, task_label_offset : task_label_offset + task_label_width
                    ]
                else:
                    labels = label_ids[:, 0, task_label_offset]
            elif label_ids.ndim == 2:
                labels = label_ids[:, task_label_offset]
            else:
                raise RuntimeError(
                    f"label_ids has the wrong number of dimensions ({label_ids.ndim})"
                )

            if labels is not None:
                labels = labels.squeeze()

            task_predictions.append(
                TaskEvalPrediction(
                    task=task,
                    predictions=preds,
                    probs=probs,
                    labels=labels,
                )
            )

            task_label_offset += task_label_width

        return task_predictions

    def _compute_metrics(self, p: EvalPrediction):
        summary_metrics = {
            "avg_acc": 0,
            "avg_micro_f1": 0,
            "avg_macro_f1": 0,
        }

        metrics: dict[str, float] = {}

        for task_prediction in self._extract_task_predictions(p):
            task_metrics = task_prediction.compute_metrics()
            metrics |= task_metrics
            for m in summary_metrics:
                summary_metrics[m] += task_metrics[
                    f"{task_prediction.task.name}.{m.removeprefix('avg_')}"
                ]

        for m in summary_metrics:
            summary_metrics[m] /= len(self.dataset.tasks)

        result = summary_metrics | metrics

        requested_metric = self.args.metric_for_best_model
        if requested_metric is not None and requested_metric not in result:
            submetrics: list[float] = []
            for sub in requested_metric.split(","):
                sub = sub.removeprefix("eval_".strip())
                if sub not in result:
                    raise ValueError(f"unknown evaluation metric {sub}")
                submetrics.append(result[sub])

            result[requested_metric] = sum(submetrics) / len(submetrics)

        return result

    @contextlib.contextmanager
    def _trainer(self):
        configure_logger_for_training(self.args)

        trainer_callbacks: list[TrainerCallback] = [BasicLoggingCallback(self)]
        if self.args.rich_display and self.args.local_rank in (-1, 0):
            self.args.disable_tqdm = True
            self.disp = TrainSystemDisplay(self)
            display_callback = DisplayCallback(self.disp)
            trainer_callbacks.append(display_callback)
        else:
            self.disp = None
            display_callback = None

        with self.disp or contextlib.nullcontext():
            trainer = Trainer(
                model=self.model,
                args=self.args,
                train_dataset=self.dataset.train_data,
                eval_dataset=self.dataset.validation_data,
                compute_metrics=self._compute_metrics,
                callbacks=trainer_callbacks,
            )
            if self.args.rich_display:
                # If tqdm is disabled, the Trainer will automatically add a PrinterCallback
                # when initialized. We disable tqdm when the rich display is active, but we
                # don't want the PrinterCallback in that case either, so we'll remove it
                # manually here.
                trainer.remove_callback(PrinterCallback)

            yield trainer

        self.disp = None

    def _evaluate(self, trainer: Trainer):
        if self.disp:
            self.disp.eval_desc = "Evaluating"
        return trainer.evaluate()

    def _predict(self, trainer: Trainer, dataset: Dataset):
        if self.disp:
            self.disp.eval_desc = "Predicting"
        raw_prediction = trainer.predict(dataset)
        return CnlpPredictions(
            dataset,
            raw_prediction,
            self.dataset.tasks,
            max_seq_length=self.dataset.max_seq_length,
        )

    def train(self):
        """Run the training loop."""

        with self._trainer() as trainer:
            if self.disp:
                self.disp.eval_desc = "Evaluating"

            trainer.train(resume_from_checkpoint=self.args.resume_from_checkpoint)
            trainer.save_model()

            if self.args.do_predict:
                predictions = self._predict(trainer, self.dataset.test_data)
                predictions_file = os.path.join(
                    self.args.output_dir, "predictions.json"
                )
                predictions.save_json(
                    predictions_file,
                    allow_overwrite=self.args.overwrite_output_dir,
                )

    def evaluate(self) -> dict[str, float]:
        """Run an evaluation on the valdiation set.

        Returns:
            Evaluation metrics.
        """

        with self._trainer() as trainer:
            return self._evaluate(trainer)

    def predict(self, dataset: Dataset | None = None) -> CnlpPredictions:
        """Run predictions on the test set.

        Args:
            dataset: Dataset to run predictions. Optional, defaults to the test data in this
                train system's dataset.

        Returns:
            The prediction output.
        """

        with self._trainer() as trainer:
            return self._predict(trainer, dataset or self.dataset.test_data)

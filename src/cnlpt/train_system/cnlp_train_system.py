import contextlib
import math
import os
from collections import Counter
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt
import torch
from datasets import Dataset
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import PrinterCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction, IntervalStrategy, set_seed

from ..args import (
    CnlpDataArguments,
    CnlpModelArguments,
    CnlpTrainingArguments,
    parse_args_dict,
    parse_args_from_argv,
    parse_args_json_file,
    preprocess_args,
)
from ..data import RELATIONS, TAGGING, CnlpDataset, CnlpPredictions
from ..models import CnlpConfig, CnlpModelForClassification, HierarchicalModel
from ..models.baseline import CnnSentenceClassifier, LstmSentenceClassifier
from .display import TrainSystemDisplay
from .log import configure_logger_for_training, logger
from .metrics import TaskEvalPrediction
from .training_callbacks import BasicLoggingCallback, DisplayCallback
from .utils import is_external_encoder, simple_softmax


class CnlpTrainSystem:
    """This class manages the full training workflow for the cnlp_transformers library.

    The train system can be initialized directly from `CnlpModelArguments`, `CnlpDataArguments`, and `CnlpTrainingArguments`,
    or using one of the following class methods:
    - `from_json_args(json_file)`: Load arguments from a json file.
    - `from_args_dict(args)`: Load arguments from a python dictionary.
    - `from_argv(argv)`: Load arguments from `sys.argv` or a user-specified list of argv-style arguments.
    """

    def __init__(
        self,
        *,
        model_args: CnlpModelArguments,
        data_args: CnlpDataArguments,
        training_args: CnlpTrainingArguments,
    ):
        configure_logger_for_training(training_args)
        preprocess_args(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.disp: Union[TrainSystemDisplay, None] = None

        set_seed(self.training_args.seed)

        self.tokenizer = self._init_tokenizer()
        self.dataset = self._init_dataset()
        self.model = self._init_model()

        self._set_eval_strategy()

    @classmethod
    def from_json_args(cls, json_file: Union[str, os.PathLike]):
        """Instantiate the train system from a json-formatted args file.

        Args:
            json_file: Path to the json-formatted args file.

        Returns:
            The new `CnlpTrainSystem` instance.
        """
        model_args, data_args, training_args = parse_args_json_file(json_file)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @classmethod
    def from_args_dict(cls, args: dict[str, Any]):
        """Instantiate the train system from a dict of args.

        Args:
            args: Arguments for the train system.

        Returns:
            The new `CnlpTrainSystem` instance.
        """
        model_args, data_args, training_args = parse_args_dict(args)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @classmethod
    def from_argv(cls, argv: Union[list[str], None] = None):
        """Instantiate the train system from `sys.argv` or a user-specified list of argv-style arguments.

        If `argv` is not specified, `sys.argv` will be used.

        Args:
            argv: List of arguments. Optional, defaults to None.

        Returns:
            The new `CnlpTrainSystem` instance.
        """
        model_args, data_args, training_args = parse_args_from_argv(argv)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    def _init_tokenizer(self):
        tokenizer_name = self.model_args.tokenizer_name or self.model_args.encoder_name
        assert tokenizer_name is not None
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.model_args.cache_dir,
            add_prefix_space=True,
            truncation_side=(
                "left" if self.training_args.truncation_side_left else "right"
            ),
            additional_special_tokens=(
                ["<e>", "</e>", "<a1>", "</a1>", "<a2>", "</a2>", "<cr>", "<neg>"]
                if not self.data_args.character_level
                else None
            ),
        )
        return cast(PreTrainedTokenizer, tokenizer)

    def _init_dataset(self):
        return CnlpDataset(
            self.data_args,
            tokenizer=self.tokenizer,
            hierarchical=(self.model_args.model == "hier"),
        )

    def _init_model(self):
        model_name = self.model_args.model
        if model_name == "cnn":
            return self._init_cnn_model()
        elif model_name == "lstm":
            return self._init_lstm_model()
        elif model_name == "hier":
            return self._init_hier_model()
        else:
            return self._init_cnlpt_model()

    def _get_class_weights(self):
        if not self.data_args.weight_classes:
            return None

        class_weights: list[list[float]] = []
        for task in self.dataset.tasks:
            train_labels = self.dataset.train_data[task.name]
            weights: list[float] = []
            train_label_counts = Counter(train_labels)
            for label in task.labels:
                # class weights are determined by severity of class imbalance
                weights.append(
                    len(train_labels) / (len(task.labels) * train_label_counts[label])
                )

            class_weights.append(weights)

        class_weights_tensor = torch.tensor(
            # if we just have the one class, simplify the tensor or pytorch will be mad
            class_weights[0] if len(class_weights) == 1 else class_weights
        ).to(self.training_args.device)

        return class_weights_tensor

    def _init_cnn_model(self):
        model = CnnSentenceClassifier(
            len(self.tokenizer),
            task_names=[t.name for t in self.dataset.tasks],
            num_labels_dict={t.name: len(t.labels) for t in self.dataset.tasks},
            embed_dims=self.model_args.cnn_embed_dim,
            num_filters=self.model_args.cnn_num_filters,
            filters=self.model_args.cnn_filter_sizes,
            use_prior_tasks=self.model_args.use_prior_tasks,
            class_weights=self._get_class_weights(),
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        assert self.model_args.encoder_name is not None
        model_path = os.path.join(self.model_args.encoder_name, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

        return model

    def _init_lstm_model(self):
        model = LstmSentenceClassifier(
            len(self.tokenizer),
            task_names=[t.name for t in self.dataset.tasks],
            num_labels_dict={t.name: len(t.labels) for t in self.dataset.tasks},
            embed_dims=self.model_args.lstm_embed_dim,
            hidden_size=self.model_args.lstm_hidden_size,
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        assert self.model_args.encoder_name is not None
        model_path = os.path.join(self.model_args.encoder_name, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

        return model

    def _init_hier_model(self):
        encoder_name = (
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.encoder_name
        )
        assert encoder_name is not None
        if is_external_encoder(encoder_name):
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=[t.name for t in self.dataset.tasks],
                layer=self.model_args.layer or -1,
                tokens=self.model_args.token,
                num_rel_attention_heads=self.model_args.num_rel_feats,
                rel_attention_head_dims=self.model_args.head_features,
                tagger={t.name: t.type == TAGGING for t in self.dataset.tasks},
                relations={t.name: t.type == RELATIONS for t in self.dataset.tasks},
                label_dictionary={t.name: list(t.labels) for t in self.dataset.tasks},
                hier_head_config=dict(
                    n_layers=self.model_args.hier_num_layers,
                    d_inner=self.model_args.hier_hidden_dim,
                    n_head=self.model_args.hier_n_head,
                    d_k=self.model_args.hier_d_k,
                    d_v=self.model_args.hier_d_v,
                    dropout=self.model_args.hier_dropout,
                ),
            )
            # num_tokens=len(tokenizer))
            config.vocab_size = len(self.tokenizer)

            model = HierarchicalModel(
                config=config,
                # TODO(ian) as far as I can tell, this was always just None?
                class_weights=None,
                freeze=self.training_args.freeze,
            )
        else:
            if (
                self.model_args.keep_existing_classifiers
                == self.model_args.ignore_existing_classifiers
            ):  # XNOR
                raise ValueError(
                    "For continued training of a cnlpt hierarchical model, one of --keep_existing_classifiers or --ignore_existing_classifiers flags should be selected."
                )
            # use a checkpoint from an existing model

            config: CnlpConfig = AutoConfig.from_pretrained(
                encoder_name,
                cache_dir=self.model_args.cache_dir,
                layer=self.model_args.layer,
            )
            task_is_relations = {
                t.name: t.type == RELATIONS for t in self.dataset.tasks
            }
            task_is_tagging = {t.name: t.type == TAGGING for t in self.dataset.tasks}

            if self.model_args.ignore_existing_classifiers:
                config.finetuning_task = [t.name for t in self.dataset.tasks]
                config.relations = task_is_relations
                config.tagger = task_is_tagging
                config.label_dictionary = {}  # this gets filled in later
            elif self.model_args.keep_existing_classifiers:
                if (
                    config.finetuning_task != [t.name for t in self.dataset.tasks]
                    or config.relations != task_is_relations
                    or config.tagger != task_is_tagging
                ):
                    raise ValueError(
                        "When --keep_existing_classifiers selected, please ensure"
                        "that you set the settings the same as those used in the"
                        "previous training run."
                    )

            # TODO: check if user overwrote parameters in command line that could change behavior of the model and warn
            # if self.data_args.chunk_len is not None:

            logger.info("Loading pre-trained hierarchical model...")
            model: HierarchicalModel = AutoModel.from_pretrained(
                encoder_name, config=config
            )

            if self.model_args.ignore_existing_classifiers:
                model.remove_task_classifiers()
                for task in self.dataset.tasks:
                    model.add_task_classifier(task.name, list(task.labels))

            # TODO(ian) as far as I can tell, this was always just None?
            model.set_class_weights(None)

        return cast(HierarchicalModel, model)

    def _init_cnlpt_model(self):
        # by default cnlpt model, but need to check which encoder they want
        encoder_name = self.model_args.encoder_name
        assert encoder_name is not None

        # TODO check when download any pretrained language model to local disk, if
        # the following condition "is_hub_model(encoder_name)" works or not.
        if not is_external_encoder(encoder_name):
            # we are loading one of our own trained models as a starting point.
            #
            # 1) if training_args.do_train is true:
            # sometimes we may want to use an encoder that has had continued pre-training, either on
            # in-domain MLM or another task we think might be useful. In that case our encoder will just
            # be a link to a directory. If the encoder-name is not recognized as a pre-trained model, special
            # logic for ad hoc encoders follows:
            # we will load it as-is initially, then delete its classifier head, save the encoder
            # as a temp file, and make that temp file
            # the model file to be loaded down below the normal way. since that temp file
            # doesn't have a stored classifier it will use the randomly-inited classifier head
            # with the size of the supplied config (for the new task).
            # TODO This setting 1) is not tested yet.
            # 2) if training_args.do_train is false:
            # we evaluate or make predictions of our trained models.
            # Both two setting require the registeration of CnlpConfig, and use
            # AutoConfig.from_pretrained() to load the configuration file

            # Load the cnlp configuration using AutoConfig, this will not override
            # the arguments from trained cnlp models. While using CnlpConfig will override
            # the model_type and model_name of the encoder.
            encoder_name = (
                self.model_args.config_name
                if self.model_args.config_name
                else encoder_name
            )
            config = AutoConfig.from_pretrained(
                encoder_name,
                cache_dir=self.model_args.cache_dir,
                # in this case we're looking at a fine-tuned model (?)
                character_level=self.data_args.character_level,
            )

            if self.training_args.do_train:
                # Setting 1) only load weights from the encoder
                raise NotImplementedError(
                    "This functionality has not been restored yet"
                )
            else:
                # setting 2) evaluate or make predictions
                model = CnlpModelForClassification.from_pretrained(
                    self.model_args.encoder_name,
                    config=config,
                    # TODO(ian) as far as I can tell, this was always just None?
                    class_weights=None,
                    final_task_weight=self.training_args.final_task_weight,
                    freeze=self.training_args.freeze,
                    bias_fit=self.training_args.bias_fit,
                )
        else:
            # This only works when model_args.encoder_name is one of the
            # model card from https://huggingface.co/models
            # By default, we use model card as the starting point to fine-tune
            encoder_name = (
                self.model_args.config_name
                if self.model_args.config_name
                else encoder_name
            )
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=[t.name for t in self.dataset.tasks],
                layer=self.model_args.layer,
                tokens=self.model_args.token,
                num_rel_attention_heads=self.model_args.num_rel_feats,
                rel_attention_head_dims=self.model_args.head_features,
                tagger={t.name: t.type == TAGGING for t in self.dataset.tasks},
                relations={t.name: t.type == RELATIONS for t in self.dataset.tasks},
                label_dictionary={t.name: list(t.labels) for t in self.dataset.tasks},
                character_level=self.data_args.character_level,
                # num_tokens=len(tokenizer),
            )
            config.vocab_size = len(self.tokenizer)
            model = CnlpModelForClassification(
                config=config,
                # TODO(ian) as far as I can tell, this was always just None?
                class_weights=None,
                final_task_weight=self.training_args.final_task_weight,
                freeze=self.training_args.freeze,
                bias_fit=self.training_args.bias_fit,
            )

        return cast(CnlpModelForClassification, model)

    def _set_eval_strategy(self):
        if not self.training_args.do_train:
            return

        batches_per_epoch = math.ceil(
            len(self.dataset.train_data) / self.training_args.train_batch_size
        )
        total_steps = int(
            self.training_args.num_train_epochs
            * batches_per_epoch
            // self.training_args.gradient_accumulation_steps
        )

        if self.training_args.evals_per_epoch > 0:
            logger.warning(
                "Overwriting the value of logging steps based on provided evals_per_epoch argument"
            )
            # steps per epoch factors in gradient accumulation steps (as compared to batches_per_epoch above which doesn't)
            steps_per_epoch = int(total_steps // self.training_args.num_train_epochs)
            self.training_args.eval_steps = (
                steps_per_epoch // self.training_args.evals_per_epoch
            )
            self.training_args.eval_strategy = self.training_args.eval_strategy = (
                IntervalStrategy.STEPS
            )
            # This will save model per epoch
            # training_args.save_strategy = IntervalStrategy.EPOCH
        elif self.training_args.do_eval:
            logger.info("Evaluation strategy not specified so evaluating every epoch")
            self.training_args.eval_strategy = self.training_args.eval_strategy = (
                IntervalStrategy.EPOCH
            )

    def _extract_task_predictions(self, p: EvalPrediction):
        task_predictions: list[TaskEvalPrediction] = []
        task_label_offset = 0

        for task in self.dataset.tasks:
            probs: Union[npt.NDArray[np.float64], None] = None

            raw_preds = p.predictions[task.index]
            if task.type == TAGGING:
                preds = np.argmax(raw_preds, axis=2)
                # labels will be -100 where we don't need to tag
            elif task.type == RELATIONS:
                preds = np.argmax(raw_preds, axis=3)
            else:
                preds = np.argmax(raw_preds, axis=1)
                if self.training_args.output_prob:
                    probs = np.max(
                        [simple_softmax(logits) for logits in raw_preds],
                        axis=1,
                    )

            labels: Union[npt.NDArray[np.int64], None]
            task_label_width = 0

            label_ids: npt.NDArray[np.int64] | None = getattr(p, "label_ids", None)
            if label_ids is None:
                # we are doing inference, so no labels
                labels = None
            elif task.type == RELATIONS:
                task_label_width = self.data_args.max_seq_length
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

        result = summary_metrics | metrics

        requested_metric = self.training_args.metric_for_best_model
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
    def trainer(self):
        trainer_callbacks: list[TrainerCallback] = [
            BasicLoggingCallback(self.model_args, self.data_args, self.training_args)
        ]

        if self.training_args.rich_display and self.training_args.local_rank in (-1, 0):
            self.training_args.disable_tqdm = True
            self.disp = TrainSystemDisplay(
                self.model_args,
                self.data_args,
                self.training_args,
            )
            display_callback = DisplayCallback(self.disp)
            trainer_callbacks.append(display_callback)
        else:
            display_callback = None

        with self.disp or contextlib.nullcontext():
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.dataset.train_data,
                eval_dataset=self.dataset.validation_data,
                compute_metrics=self._compute_metrics,
                callbacks=trainer_callbacks,
            )
            if self.training_args.rich_display:
                # remove the PrinterCallback added by default when we initialized the trainer
                trainer.remove_callback(PrinterCallback)

            yield trainer

        self.disp = None

    def train(self):
        """Begin the training loop."""

        with self.trainer() as trainer:
            if self.disp:
                self.disp.eval_desc = "Evaluating"

            trainer.train()
            trainer.save_model()

            if self.training_args.do_predict:
                predictions = self._predict(trainer, self.dataset.test_data)
                predictions_file = os.path.join(
                    self.training_args.output_dir, "predictions.json"
                )
                predictions.save_json(
                    predictions_file,
                    allow_overwrite=self.training_args.overwrite_output_dir,
                )

    def _evaluate(self, trainer: Trainer):
        if self.disp:
            self.disp.eval_desc = "Evaluating"
        return trainer.evaluate()

    def _predict(self, trainer: Trainer, dataset: Dataset):
        if self.disp:
            self.disp.eval_desc = "Predicting"
        raw_prediction = trainer.predict(dataset)
        return CnlpPredictions(
            dataset, raw_prediction, self.dataset.tasks, self.data_args
        )

    def evaluate(self) -> dict[str, float]:
        """Run an evaluation on the valdiation set.

        Returns:
            Evaluation metrics.
        """

        with self.trainer() as trainer:
            return self._evaluate(trainer)

    def predict(self, dataset: Union[Dataset, None] = None) -> CnlpPredictions:
        """Run predictions on the test set.

        Args:
            dataset: Dataset to run predictions. Optional, defaults to the test data in this
            train system's dataset.

        Returns:
            The prediction output.
        """

        with self.trainer() as trainer:
            return self._predict(trainer, dataset or self.dataset.test_data)


def main(argv: Union[list[str], None] = None):
    train_system = CnlpTrainSystem.from_argv(argv)
    train_system.train()

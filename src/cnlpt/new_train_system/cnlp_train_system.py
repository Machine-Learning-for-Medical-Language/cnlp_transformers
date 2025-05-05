import contextlib
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Union, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.trainer_callback import PrinterCallback, TrainerCallback
from transformers.trainer_utils import EvalPrediction, IntervalStrategy, set_seed

from ..args import CnlpDataArguments, CnlpModelArguments, CnlpTrainingArguments
from ..models import CnlpConfig, CnlpModelForClassification, HierarchicalModel
from ..models.baseline import CnnSentenceClassifier, LstmSentenceClassifier
from ..new_data.cnlp_dataset import CnlpDataset
from ..new_data.task_info import RELATIONS, TAGGING, TaskInfo
from .display import TrainSystemDisplay
from .logging import configure_logger_for_training, logger
from .metrics import cnlp_compute_metrics
from .parse_args import (
    parse_args_dict,
    parse_args_from_argv,
    parse_args_json_file,
    validate_args,
)
from .training_callbacks import (
    BasicLoggingCallback,
    DisplayCallback,
    SaveBestModelCallback,
)
from .utils import is_external_encoder, simple_softmax


@dataclass(frozen=True)
class TaskEvalPrediction:
    task: TaskInfo
    predictions: np.ndarray
    probs: Union[np.ndarray, None]
    labels: Union[np.ndarray, None]


class CnlpTrainSystem:
    def __init__(
        self,
        *,
        model_args: CnlpModelArguments,
        data_args: CnlpDataArguments,
        training_args: CnlpTrainingArguments,
    ):
        configure_logger_for_training(training_args)
        validate_args(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        del model_args, data_args, training_args

        set_seed(self.training_args.seed)
        self._init_tokenizer()
        self._init_dataset()
        self._init_model()
        self._preprocess_training_args()

        # TODO (in new eval() method)
        # - evaluation

        # TODO (in new predict() method)
        # - prediction

        # TODO run tests with new system

    @classmethod
    def from_json_args(cls, json_file: Union[str, os.PathLike]):
        model_args, data_args, training_args = parse_args_json_file(json_file)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @classmethod
    def from_args_dict(cls, args: dict[str, Any]):
        model_args, data_args, training_args = parse_args_dict(args)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @classmethod
    def from_argv(cls, argv: Union[list[str], None] = None):
        model_args, data_args, training_args = parse_args_from_argv(argv)
        return cls(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    def _init_tokenizer(self):
        tokenizer_name = self.model_args.tokenizer_name or self.model_args.encoder_name
        assert tokenizer_name is not None
        self.tokenizer = AutoTokenizer.from_pretrained(
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

    def _init_dataset(self):
        self.dataset = CnlpDataset(
            self.data_args,
            tokenizer=self.tokenizer,
            hierarchical=(self.model_args.model == "hier"),
        )

    def _init_model(self):
        model_name = self.model_args.model
        if model_name == "cnn":
            self._init_cnn_model()
        elif model_name == "lstm":
            self._init_lstm_model()
        elif model_name == "hier":
            self._init_hier_model()
        else:
            self._init_cnlpt_model()

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

        self.model = cast(CnnSentenceClassifier, model)

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

        self.model = cast(LstmSentenceClassifier, model)

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

        self.model = cast(HierarchicalModel, model)

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

        self.model = cast(CnlpModelForClassification, model)

    def _preprocess_training_args(self):
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

        if torch.mps.is_available():
            # pin_memory is unsupported on MPS, but defaults to True,
            # so we'll explicitly turn it off to avoid a warning.
            self.training_args.dataloader_pin_memory = False

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

    def _get_task_prediction_metrics(self, task_prediction: TaskEvalPrediction):
        if task_prediction.labels is None:
            raise RuntimeError(
                "cannot compute metrics because eval prediction has no labels"
            )
        return cnlp_compute_metrics(
            task_prediction.predictions,
            task_prediction.labels,
            task_prediction.task.type,
            list(task_prediction.task.labels),
        )

    def _compute_metrics(self, p: EvalPrediction):
        metrics: dict[str, dict[str, Any]] = {}

        for task_prediction in self._extract_task_predictions(p):
            task = task_prediction.task
            metrics[task.name] = self._get_task_prediction_metrics(task_prediction)

        return metrics

    def train(self, rich_display: bool = True):
        trainer_callbacks: list[TrainerCallback] = [
            BasicLoggingCallback(self.model_args, self.data_args, self.training_args)
        ]

        selection_labels = self.training_args.model_selection_label
        if not isinstance(selection_labels, list):
            selection_labels = [selection_labels]
        selection_labels = [str(label) for label in selection_labels]

        assert self.training_args.output_dir is not None
        save_best_model_callback = SaveBestModelCallback(
            tasks=self.dataset.tasks,
            selection_metric=self.training_args.model_selection_score,
            selection_labels=selection_labels,
        )

        trainer_callbacks.append(save_best_model_callback)

        if rich_display:
            self.training_args.disable_tqdm = True
            disp = TrainSystemDisplay(
                self.model_args,
                self.data_args,
                self.training_args,
            )
            trainer_callbacks.append(DisplayCallback(disp, save_best_model_callback))
        else:
            disp = contextlib.nullcontext()

        with disp:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.dataset.train_data,
                eval_dataset=self.dataset.validation_data,
                compute_metrics=self._compute_metrics,
                callbacks=trainer_callbacks,
            )

            if rich_display:
                trainer.remove_callback(PrinterCallback)

            trainer.train()

    def evaluate(self):
        # TODO(ian)
        # This method should just run a single evaluation (on the validation data) with the current model.
        pass

    def predict(self):
        # TODO(ian)
        # This method should run predictions on the test data with the current model.
        pass


def main(argv: Union[list[str], None] = None):
    train_system = CnlpTrainSystem.from_argv(argv)
    train_system.train()

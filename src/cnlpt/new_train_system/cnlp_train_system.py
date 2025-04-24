import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    IntervalStrategy,
    set_seed,
)

from ..args import CnlpTrainingArguments, DataTrainingArguments, ModelArguments
from ..data.cnlp_datasets import ClinicalNlpDataset
from ..data.tasks import RELEX, TAGGING
from ..models import CnlpConfig, CnlpModelForClassification, HierarchicalModel
from ..models.baseline import CnnSentenceClassifier, LstmSentenceClassifier
from .args import (
    parse_args_dict,
    parse_args_from_argv,
    parse_args_json_file,
    validate_args,
)
from .log import configure_logger_for_training, logger
from .utils import is_external_encoder


@dataclass
class _TaskMaps:
    task_names: list[str]
    num_labels: dict[str, int]
    output_mode: dict[str, str]
    tagger: dict[str, bool]
    relations: dict[str, bool]


class CnlpTrainSystem:
    def __init__(
        self,
        *,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
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

        # TODO compute metrics function

        # TODO (in new train() method)
        # - init trainer
        # - train

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            (
                self.model_args.tokenizer_name
                if self.model_args.tokenizer_name
                else self.model_args.encoder_name
            ),
            cache_dir=self.model_args.cache_dir,
            add_prefix_space=True,
            truncation_side="left"
            if self.training_args.truncation_side_left
            else "right",
            additional_special_tokens=(
                [
                    "<e>",
                    "</e>",
                    "<a1>",
                    "</a1>",
                    "<a2>",
                    "</a2>",
                    "<cr>",
                    "<neg>",
                ]
                if not self.data_args.character_level
                else None
            ),
        )

    def _init_dataset(self):
        self.dataset = ClinicalNlpDataset(
            self.data_args,
            tokenizer=self.tokenizer,
            cache_dir=self.model_args.cache_dir,
            hierarchical=(self.model_args.model == "hier"),
        )
        try:
            task_names = (
                self.data_args.task_name
                if self.data_args.task_name is not None
                else self.dataset.tasks
            )
            num_labels: dict[str, int] = {}
            output_mode: dict[str, str] = {}
            tagger: dict[str, bool] = {}
            relations: dict[str, bool] = {}
            for task in self.dataset.tasks_to_labels.keys():
                num_labels[task] = len(self.dataset.tasks_to_labels[task])
                task_output_mode: str = self.dataset.output_modes[task]
                output_mode[task] = task_output_mode
                tagger[task] = task_output_mode == TAGGING
                relations[task] = task_output_mode == RELEX
        except KeyError:
            raise ValueError(f"Task not found: {self.data_args.task_name}")

        self.task_maps = _TaskMaps(
            task_names, num_labels, output_mode, tagger, relations
        )

    def _init_model(self):
        # Load pretrained model
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

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
        class_weights = []
        for task in self.task_maps.task_names:
            # get labels in the right order ([0, 1])
            if isinstance(
                self.dataset.tasks_to_labels[task][1], str
            ) and self.dataset.tasks_to_labels[task][1].startswith("No_"):
                self.dataset.tasks_to_labels[task] = self.dataset.tasks_to_labels[task][
                    1:
                ] + [self.dataset.tasks_to_labels[task][0]]
            labels = self.dataset.processed_dataset["train"][task]
            weights = []
            label_counts = Counter(labels)
            for label in self.dataset.tasks_to_labels[task]:
                weights.append(
                    len(labels)
                    / (self.task_maps.num_labels[task] * label_counts[label])
                )
                # class weights are determined by severity of class imbalance
            if len(self.task_maps.task_names) > 1:
                class_weights.append(weights)
            else:
                class_weights = weights  # if we just have the one class, simplify the tensor or pytorch will be mad
        class_weights = torch.tensor(class_weights).to(self.training_args.device)
        # sm = torch.nn.Softmax(dim=class_weights.ndim - 1)
        # class_weights = sm(class_weights)
        return class_weights

    def _init_cnn_model(self):
        model = CnnSentenceClassifier(
            len(self.tokenizer),
            task_names=self.task_maps.task_names,
            num_labels_dict=self.task_maps.num_labels,
            embed_dims=self.model_args.cnn_embed_dim,
            num_filters=self.model_args.cnn_num_filters,
            filters=self.model_args.cnn_filter_sizes,
            use_prior_tasks=self.model_args.use_prior_tasks,
            class_weights=self._get_class_weights(),
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = os.path.join(self.model_args.encoder_name, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

        self.model = model

    def _init_lstm_model(self):
        model = LstmSentenceClassifier(
            len(self.tokenizer),
            task_names=self.task_maps.task_names,
            num_labels_dict=self.task_maps.num_labels,
            embed_dims=self.model_args.lstm_embed_dim,
            hidden_size=self.model_args.lstm_hidden_size,
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = os.path.join(self.model_args.encoder_name, "pytorch_model.bin")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

        self.model = model

    def _init_hier_model(self):
        encoder_name = (
            self.model_args.config_name
            if self.model_args.config_name
            else self.model_args.encoder_name
        )
        if is_external_encoder(encoder_name):
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=(
                    self.data_args.task_name
                    if self.data_args.task_name is not None
                    else self.dataset.tasks
                ),
                layer=self.model_args.layer,
                tokens=self.model_args.token,
                num_rel_attention_heads=self.model_args.num_rel_feats,
                rel_attention_head_dims=self.model_args.head_features,
                tagger=self.task_maps.tagger,
                relations=self.task_maps.relations,
                label_dictionary=self.dataset.get_labels(),
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
                class_weights=self.dataset.class_weights,
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

            config = AutoConfig.from_pretrained(
                encoder_name,
                cache_dir=self.model_args.cache_dir,
                layer=self.model_args.layer,
            )
            if self.model_args.ignore_existing_classifiers:
                config.finetuning_task = (
                    self.data_args.task_name
                    if self.data_args.task_name is not None
                    else self.dataset.tasks
                )
                config.relations = self.task_maps.relations
                config.tagger = self.task_maps.tagger
                config.label_dictionary = {}  # this gets filled in later
            elif self.model_args.keep_existing_classifiers:
                if (
                    config.finetuning_task != self.data_args.task_name
                    or config.relations != self.task_maps.relations
                    or config.tagger != self.task_maps.tagger
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
                for task in self.data_args.task_name:
                    model.add_task_classifier(task, self.dataset.get_labels()[task])
            model.set_class_weights(self.dataset.class_weights)

        self.model = model

    def _init_cnlpt_model(self):
        # by default cnlpt model, but need to check which encoder they want
        encoder_name = self.model_args.encoder_name

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
            config = AutoConfig.from_pretrained(
                (
                    self.model_args.config_name
                    if self.model_args.config_name
                    else self.model_args.encoder_name
                ),
                cache_dir=self.model_args.cache_dir,
                # in this case we're looking at a fine-tuned model (?)
                character_level=self.data_args.character_level,
            )

            if self.training_args.do_train:
                # Setting 1) only load weights from the encoder
                raise NotImplementedError(
                    "This functionality has not been restored yet"
                )
                # model = CnlpModelForClassification(
                #     model_path=self.model_args.encoder_name,
                #     config=config,
                #     cache_dir=self.model_args.cache_dir,
                #     tagger=self.task_maps.tagger,
                #     relations=self.task_maps.relations,
                #     class_weights=self.dataset.class_weights,
                #     final_task_weight=self.training_args.final_task_weight,
                #     use_prior_tasks=self.model_args.use_prior_tasks,
                #     argument_regularization=self.model_args.arg_reg,
                # )
                # delattr(model, "classifiers")
                # delattr(model, "feature_extractors")
                # if self.training_args.do_train:
                #     tempmodel = tempfile.NamedTemporaryFile(
                #         dir=self.model_args.cache_dir
                #     )
                #     torch.save(model.state_dict(), tempmodel)
                #     model_name = tempmodel.name
            else:
                # setting 2) evaluate or make predictions
                model = CnlpModelForClassification.from_pretrained(
                    self.model_args.encoder_name,
                    config=config,
                    class_weights=self.dataset.class_weights,
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
                else self.model_args.encoder_name
            )
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=(
                    self.data_args.task_name
                    if self.data_args.task_name is not None
                    else self.dataset.tasks
                ),
                layer=self.model_args.layer,
                tokens=self.model_args.token,
                num_rel_attention_heads=self.model_args.num_rel_feats,
                rel_attention_head_dims=self.model_args.head_features,
                tagger=self.task_maps.tagger,
                relations=self.task_maps.relations,
                label_dictionary=self.dataset.get_labels(),
                character_level=self.data_args.character_level,
                # num_tokens=len(tokenizer),
            )
            config.vocab_size = len(self.tokenizer)
            model = CnlpModelForClassification(
                config=config,
                class_weights=self.dataset.class_weights,
                final_task_weight=self.training_args.final_task_weight,
                freeze=self.training_args.freeze,
                bias_fit=self.training_args.bias_fit,
            )

        self.model = model

    def _preprocess_training_args(self):
        if not self.training_args.do_train:
            return

        batches_per_epoch = math.ceil(
            self.dataset.num_train_instances / self.training_args.train_batch_size
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
            self.training_args.evaluation_strategy = (
                self.training_args.eval_strategy
            ) = IntervalStrategy.STEPS
            # This will save model per epoch
            # training_args.save_strategy = IntervalStrategy.EPOCH
        elif self.training_args.do_eval:
            logger.info("Evaluation strategy not specified so evaluating every epoch")
            self.training_args.evaluation_strategy = (
                self.training_args.eval_strategy
            ) = IntervalStrategy.EPOCH

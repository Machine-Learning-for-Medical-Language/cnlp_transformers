# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless require wdd by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for sequence classification on clinical NLP tasks"""

import csv
import json
import logging
import math
import os
import sys
import tempfile
from collections import defaultdict, deque
from os.path import exists, join
from typing import Any, Callable, Union

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_url
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    set_seed,
)
from transformers.file_utils import CONFIG_NAME
from transformers.training_args import IntervalStrategy

from .BaselineModels import CnnSentenceClassifier, LstmSentenceClassifier
from .cnlp_args import CnlpTrainingArguments, ModelArguments
from .cnlp_data import ClinicalNlpDataset, DataTrainingArguments, get_dataset_segment
from .cnlp_metrics import cnlp_compute_metrics
from .cnlp_predict import process_prediction, restructure_prediction, structure_labels
from .cnlp_processors import relex, tagging
from .CnlpModelForClassification import CnlpConfig, CnlpModelForClassification
from .HierarchicalTransformer import HierarchicalModel

sys.path.append(os.path.join(os.getcwd()))


logger = logging.getLogger(__name__)


eval_state = defaultdict(lambda: -1)


# For stopping with actual_epochs while
# spoofing the lr scheduler with num_train_epochs
# as described in the README
class StopperCallback(TrainerCallback):
    """ """

    def __init__(self, last_step=-1, last_epoch=-1):
        self.last_step = last_step
        self.last_epoch = last_epoch

    def on_step_end(self, args, state, control, **kwargs):
        """

        Args:
          args:
          state:
          control:
          **kwargs:

        Returns:

        """
        control.should_training_stop = (
            self.last_epoch > 0 and state.epoch >= self.last_epoch
        ) or (self.last_step > 0 and state.global_step >= self.last_step)


eval_state = defaultdict(lambda: -1)


# For debugging early stopping logging
class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            model_dict = {}
            if "model" in kwargs:
                model = kwargs["model"]
                if (
                    hasattr(model, "best_score")
                    and model.best_score > eval_state["best_score"]
                ):
                    model_dict = {
                        "best_score": model.best_score,
                        "best_step": state.global_step,
                        "best_epoch": state.epoch,
                    }
            state_dict = {
                "curr_epoch": state.epoch,
                "max_epochs": state.num_train_epochs,
                "curr_step": state.global_step,
                "max_steps": state.max_steps,
            }
            state_dict.update(model_dict)
            eval_state.update(state_dict)


def is_hub_model(model_name: str) -> bool:
    """
    Check for whether a model specification string is on the huggingface model hub
    :param model_name: the string to check
    :return: whether the model is on the huggingface hub
    """
    try:
        url = hf_hub_url(model_name, CONFIG_NAME)
        r = requests.head(url)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    return False


def is_cnlpt_model(model_path: str) -> bool:
    """
    Infer whether a model path refers to a cnlpt
    model checkpoint (if not, we assume it is an
    encoder)
    :param model_path: the path to the model
    :return: whether the model is a cnlpt classifier model
    """
    encoder_config = AutoConfig.from_pretrained(model_path)
    return encoder_config.model_type == "cnlpt"


def is_external_encoder(model_name_or_path: str) -> bool:
    """
    Check whether a specified model is not a cnlpt model -- an external model like a
    huggingface hub model or a downloaded local directory.
    :param model_name_or_path: specified model
    :return: whether the encoder is an external (non-cnlpt) model
    """
    return not is_cnlpt_model(model_name_or_path)


def main(
    json_file: Union[str, None] = None,
    json_obj: Union[dict[str, Any], None] = None,
) -> dict[str, dict[str, Any]]:
    """
    See all possible arguments in :class:`transformers.TrainingArguments`
    or by passing the --help flag to this script.

    We now keep distinct sets of args, for a cleaner separation of concerns.

    :param json_file: if passed, a path to a JSON file
        to use as the model, data, and training arguments instead of
        retrieving them from the CLI (mutually exclusive with ``json_obj``)
    :param json_obj: if passed, a JSON dictionary
        to use as the model, data, and training arguments instead of
        retrieving them from the CLI (mutually exclusive with ``json_file``)
    :return: the evaluation results (will be empty if ``--do_eval`` not passed)
    """
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CnlpTrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: CnlpTrainingArguments

    if json_file is not None and json_obj is not None:
        raise ValueError("cannot specify json_file and json_obj")

    if json_file is not None:
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=json_file
        )
    elif json_obj is not None:
        model_args, data_args, training_args = parser.parse_dict(json_obj)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    model_name = model_args.model
    hierarchical = model_name == "hier"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    # Set seed
    set_seed(training_args.seed)

    if training_args.model_selection_label is not None and any(
        isinstance(item, int) for item in training_args.model_selection_label
    ):
        logger.warning(
            f"It is not recommended to use ints as model selection labels: {tuple([item for item in training_args.model_selection_label if isinstance(item, int)])}. Labels should be input in string form."
        )

    # Load tokenizer: Need this first for loading the datasets
    if training_args.truncation_side_left:
        if hierarchical:
            logger.warning(
                "truncation_side_left flag is not available for the hierarchical model -- setting to right"
            )
            truncation_side = "right"
        else:
            truncation_side = "left"
    else:
        truncation_side = "right"
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.encoder_name
        ),
        cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        truncation_side=truncation_side,
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
            if not data_args.character_level
            else None
        ),
    )

    # Get datasets
    dataset = ClinicalNlpDataset(
        data_args,
        tokenizer=tokenizer,
        cache_dir=model_args.cache_dir,
        hierarchical=hierarchical,
    )

    try:
        task_names = (
            data_args.task_name if data_args.task_name is not None else dataset.tasks
        )
        num_labels = {}
        output_mode = {}
        tagger = {}
        relations = {}
        for task in dataset.tasks_to_labels.keys():
            num_labels[task] = len(dataset.tasks_to_labels[task])
            task_output_mode = dataset.output_modes[task]
            output_mode[task] = task_output_mode
            tagger[task] = task_output_mode == tagging
            relations[task] = task_output_mode == relex

    except KeyError:
        raise ValueError(f"Task not found: {data_args.task_name}")

    class_weights = None
    # get class weights, if desired
    if data_args.weight_classes:
        from collections import Counter

        class_weights = []
        for task in task_names:
            # get labels in the right order ([0, 1])
            if isinstance(
                dataset.tasks_to_labels[task][1], str
            ) and dataset.tasks_to_labels[task][1].startswith("No_"):
                dataset.tasks_to_labels[task] = dataset.tasks_to_labels[task][1:] + [
                    dataset.tasks_to_labels[task][0]
                ]
            labels = dataset.processed_dataset["train"][task]
            weights = []
            label_counts = Counter(labels)
            for label in dataset.tasks_to_labels[task]:
                weights.append(len(labels) / (num_labels[task] * label_counts[label]))
                # class weights are determined by severity of class imbalance
            if len(task_names) > 1:
                class_weights.append(weights)
            else:
                class_weights = weights  # if we just have the one class, simplify the tensor or pytorch will be mad
        class_weights = torch.tensor(class_weights).to(training_args.device)
        # sm = torch.nn.Softmax(dim=class_weights.ndim - 1)
        # class_weights = sm(class_weights)

    # Load pretrained model and tokenizer

    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_name == "cnn":
        model = CnnSentenceClassifier(
            len(tokenizer),
            task_names=task_names,
            num_labels_dict=num_labels,
            embed_dims=model_args.cnn_embed_dim,
            num_filters=model_args.cnn_num_filters,
            filters=model_args.cnn_filter_sizes,
            use_prior_tasks=model_args.use_prior_tasks,
            class_weights=class_weights,
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = join(model_args.encoder_name, "pytorch_model.bin")
        if exists(model_path):
            model.load_state_dict(torch.load(model_path))
    elif model_name == "lstm":
        model = LstmSentenceClassifier(
            len(tokenizer),
            task_names=task_names,
            num_labels_dict=num_labels,
            embed_dims=model_args.lstm_embed_dim,
            hidden_size=model_args.lstm_hidden_size,
        )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = join(model_args.encoder_name, "pytorch_model.bin")
        if exists(model_path):
            model.load_state_dict(torch.load(model_path))
    elif model_name == "hier":
        encoder_name = (
            model_args.config_name
            if model_args.config_name
            else model_args.encoder_name
        )
        if is_external_encoder(encoder_name):
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=(
                    data_args.task_name
                    if data_args.task_name is not None
                    else dataset.tasks
                ),
                layer=model_args.layer,
                tokens=model_args.token,
                num_rel_attention_heads=model_args.num_rel_feats,
                rel_attention_head_dims=model_args.head_features,
                tagger=tagger,
                relations=relations,
                label_dictionary=dataset.get_labels(),
                hier_head_config=dict(
                    n_layers=model_args.hier_num_layers,
                    d_inner=model_args.hier_hidden_dim,
                    n_head=model_args.hier_n_head,
                    d_k=model_args.hier_d_k,
                    d_v=model_args.hier_d_v,
                    dropout=model_args.hier_dropout,
                ),
            )
            # num_tokens=len(tokenizer))
            config.vocab_size = len(tokenizer)

            model = HierarchicalModel(
                config=config,
                class_weights=dataset.class_weights,
                freeze=training_args.freeze,
            )
        else:
            if hierarchical and (
                model_args.keep_existing_classifiers
                == model_args.ignore_existing_classifiers
            ):  # XNOR
                raise ValueError(
                    "For continued training of a cnlpt hierarchical model, one of --keep_existing_classifiers or --ignore_existing_classifiers flags should be selected."
                )
            # use a checkpoint from an existing model

            config = AutoConfig.from_pretrained(
                encoder_name, cache_dir=model_args.cache_dir, layer=model_args.layer
            )
            if model_args.ignore_existing_classifiers:
                config.finetuning_task = (
                    data_args.task_name
                    if data_args.task_name is not None
                    else dataset.tasks
                )
                config.relations = relations
                config.tagger = tagger
                config.label_dictionary = {}  # this gets filled in later
            elif model_args.keep_existing_classifiers:
                if (
                    config.finetuning_task != data_args.task_name
                    or config.relations != relations
                    or config.tagger != tagger
                ):
                    raise ValueError(
                        "When --keep_existing_classifiers selected, please ensure"
                        "that you set the settings the same as those used in the"
                        "previous training run."
                    )

            # TODO: check if user overwrote parameters in command line that could change behavior of the model and warn
            # if data_args.chunk_len is not None:

            logger.info("Loading pre-trained hierarchical model...")
            model = AutoModel.from_pretrained(encoder_name, config=config)

            if model_args.ignore_existing_classifiers:
                model.remove_task_classifiers()
                for task in data_args.task_name:
                    model.add_task_classifier(task, dataset.get_labels()[task])
            model.set_class_weights(dataset.class_weights)

    else:
        # by default cnlpt model, but need to check which encoder they want
        encoder_name = model_args.encoder_name

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
                    model_args.config_name
                    if model_args.config_name
                    else model_args.encoder_name
                ),
                cache_dir=model_args.cache_dir,
                # in this case we're looking at a fine-tuned model (?)
                character_level=data_args.character_level,
            )

            if training_args.do_train:
                # Setting 1) only load weights from the encoder
                raise NotImplementedError(
                    "This functionality has not been restored yet"
                )
                model = CnlpModelForClassification(
                    model_path=model_args.encoder_name,
                    config=config,
                    cache_dir=model_args.cache_dir,
                    tagger=tagger,
                    relations=relations,
                    class_weights=dataset.class_weights,
                    final_task_weight=training_args.final_task_weight,
                    use_prior_tasks=model_args.use_prior_tasks,
                    argument_regularization=model_args.arg_reg,
                )
                delattr(model, "classifiers")
                delattr(model, "feature_extractors")
                if training_args.do_train:
                    tempmodel = tempfile.NamedTemporaryFile(dir=model_args.cache_dir)
                    torch.save(model.state_dict(), tempmodel)
                    model_name = tempmodel.name
            else:
                # setting 2) evaluate or make predictions
                model = CnlpModelForClassification.from_pretrained(
                    model_args.encoder_name,
                    config=config,
                    class_weights=dataset.class_weights,
                    final_task_weight=training_args.final_task_weight,
                    freeze=training_args.freeze,
                    bias_fit=training_args.bias_fit,
                )

        else:
            # This only works when model_args.encoder_name is one of the
            # model card from https://huggingface.co/models
            # By default, we use model card as the starting point to fine-tune
            encoder_name = (
                model_args.config_name
                if model_args.config_name
                else model_args.encoder_name
            )
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=(
                    data_args.task_name
                    if data_args.task_name is not None
                    else dataset.tasks
                ),
                layer=model_args.layer,
                tokens=model_args.token,
                num_rel_attention_heads=model_args.num_rel_feats,
                rel_attention_head_dims=model_args.head_features,
                tagger=tagger,
                relations=relations,
                label_dictionary=dataset.get_labels(),
                character_level=data_args.character_level,
                # num_tokens=len(tokenizer),
            )
            config.vocab_size = len(tokenizer)
            model = CnlpModelForClassification(
                config=config,
                class_weights=dataset.class_weights,
                final_task_weight=training_args.final_task_weight,
                freeze=training_args.freeze,
                bias_fit=training_args.bias_fit,
            )

    model_type = type(model)
    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    if training_args.do_train:
        # TODO: This assumes that if there are multiple training sets, they all have the same length, but
        # in the future it would be nice to be able to have multiple heterogeneous datasets
        batches_per_epoch = math.ceil(
            dataset.num_train_instances / training_args.train_batch_size
        )
        total_steps = int(
            training_args.num_train_epochs
            * batches_per_epoch
            // training_args.gradient_accumulation_steps
        )

        if training_args.evals_per_epoch > 0:
            logger.warning(
                "Overwriting the value of logging steps based on provided evals_per_epoch argument"
            )
            # steps per epoch factors in gradient accumulation steps (as compared to batches_per_epoch above which doesn't)
            steps_per_epoch = int(total_steps // training_args.num_train_epochs)
            training_args.eval_steps = steps_per_epoch // training_args.evals_per_epoch
            training_args.evaluation_strategy = training_args.eval_strategy = (
                IntervalStrategy.STEPS
            )
            # This will save model per epoch
            # training_args.save_strategy = IntervalStrategy.EPOCH
        elif training_args.do_eval:
            logger.info("Evaluation strategy not specified so evaluating every epoch")
            training_args.evaluation_strategy = training_args.eval_strategy = (
                IntervalStrategy.EPOCH
            )

    current_prediction_packet = deque()

    def build_compute_metrics_fn(
        task_names: list[str], model, dataset: ClinicalNlpDataset
    ) -> Callable[[EvalPrediction], dict]:
        def compute_metrics_fn(p: EvalPrediction):
            metrics = {}
            task_scores = []
            task_label_ind = 0

            # disagreement collection stuff for this scope
            task_label_to_boundaries = {}
            task_label_to_label_packet = {}

            for task_ind, task_name in enumerate(dataset.tasks):
                preds, labels, pad, prob_values = structure_labels(
                    p,
                    task_name,
                    task_ind,
                    task_label_ind,
                    data_args.max_seq_length,
                    tagger,
                    relations,
                    task_label_to_boundaries,
                    training_args.output_prob,
                )
                task_label_ind += pad

                task_label_to_label_packet[task_name] = (preds, labels, prob_values)

                metrics[task_name] = cnlp_compute_metrics(
                    task_name,
                    preds,
                    labels,
                    dataset.output_modes[task_name],
                    dataset.tasks_to_labels[task_name],
                )
                # FIXME - Defaulting to accuracy for model selection score, when it should be task-specific
                if training_args.model_selection_score is not None:
                    score = metrics[task_name].get(
                        "one_score",
                        metrics[task_name].get(training_args.model_selection_score),
                    )
                    # TODO handle multi-task cases where a label is present in one class but not all
                    if isinstance(training_args.model_selection_label, int):
                        task_scores.append(score[training_args.model_selection_label])
                    # we can only get the scores in list form,
                    # so we have to maneuver a bit to get the sccore
                    # if the label is provided in string form
                    elif isinstance(training_args.model_selection_label, str):
                        index = dataset.tasks_to_labels[task_name].index(
                            training_args.model_selection_label
                        )
                        task_scores.append(score[index])
                    elif isinstance(
                        training_args.model_selection_label, list
                    ) or isinstance(training_args.model_selection_label, tuple):
                        scores = []
                        for label in training_args.model_selection_label:
                            if isinstance(label, int):
                                scores.append(score[label])
                            elif isinstance(label, str):
                                index = dataset.tasks_to_labels[task_name].index(label)
                                scores.append(score[index])
                            else:
                                raise RuntimeError(
                                    f"Unrecognized label type: {type(label)}"
                                )
                        task_scores.append(np.mean(scores))
                    elif training_args.model_selection_label is None:
                        task_scores.append(
                            metrics[task_name].get("one_score", np.mean(score))
                        )
                    else:
                        raise RuntimeError(
                            f"Unrecognized label type: {type(training_args.model_selection_label)}"
                        )
                else:  # same default as in 0.6.0
                    task_scores.append(
                        metrics[task_name].get(
                            "one_score", np.mean(metrics[task_name].get("f1"))
                        )
                    )
                # task_scores.append(processor.get_one_score(metrics.get(task_name, metrics.get(task_name.split('-')[0], None))))

            one_score = sum(task_scores) / len(task_scores)

            if model is not None:
                if not hasattr(model, "best_score") or one_score > model.best_score:
                    # For convenience, we also re-save the tokenizer to the same directory,
                    # so that you can share your model easily on huggingface.co/models =)

                    model.best_score = one_score
                    model.best_eval_results = metrics
                    if trainer.is_world_process_zero():
                        if training_args.do_train:
                            trainer.save_model()
                            tokenizer.save_pretrained(training_args.output_dir)
                            if model_name == "cnn" or model_name == "lstm":
                                with open(
                                    os.path.join(
                                        training_args.output_dir, "config.json"
                                    ),
                                    "w",
                                ) as f:
                                    config_dict = model_args.to_dict()
                                    config_dict["label_dictionary"] = (
                                        dataset.get_labels()
                                    )
                                    config_dict["task_names"] = task_names
                                    json.dump(config_dict, f)
                        for task_ind, task_name in enumerate(metrics):
                            with open(output_eval_file, "a") as writer:
                                logger.info(
                                    f"***** Eval results for task {task_name} *****"
                                )
                                writer.write(
                                    f"\n\n***** Eval results for task {task_name} *****\n\n"
                                )
                                for key, value in metrics[task_name].items():
                                    logger.info("  %s = %s", key, value)
                                    writer.write(f"{key} = {value}\n")

                    if training_args.error_analysis:
                        if len(current_prediction_packet) > 0:
                            current_prediction_packet.pop()
                        # in theory if we can consolidate this into
                        # cnlp_compute_metrics but that's maybe more of a
                        # commitment than is a good idea right now
                        current_prediction_packet.append(
                            (
                                task_label_to_label_packet,
                                task_label_to_boundaries,
                            )
                        )
            metrics["one_score"] = one_score
            return metrics

        return compute_metrics_fn

    # Initialize our Trainer
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "one_score"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.processed_dataset.get("train", None),
        eval_dataset=dataset.processed_dataset.get("validation", None),
        compute_metrics=build_compute_metrics_fn(task_names, model, dataset),
        callbacks=[EvalCallback],
    )

    # Training
    if training_args.do_train:
        trainer.train(
            # resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )

        # if we didn't do any evaluations during training then no model
        # would have ever been saved. we'll save the model here
        if not hasattr(model, "best_score"):
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                trainer.save_model()
                tokenizer.save_pretrained(training_args.output_dir)
                if model_name == "cnn" or model_name == "lstm":
                    config_dict = model_args.to_dict()
                    config_dict["label_dictionary"] = dataset.get_labels()
                    config_dict["task_names"] = task_names
                    with open(
                        os.path.join(training_args.output_dir, "config.json"), "w"
                    ) as f:
                        json.dump(config_dict, f)

    # Evaluation
    eval_results = {}

    task_to_label_space = dataset.get_labels()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_dataset = dataset.processed_dataset["validation"]
        # no evaluation was done prior to now, so we need to evaluate
        if not hasattr(model, "best_eval_results"):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        else:
            eval_result = model.best_eval_results

        # if there is a stored model, restore it so writing outputs uses a good model

        curr_step = 0
        trainer.compute_metrics = None
        if trainer.is_world_process_zero():
            # with open(output_eval_file, "w") as writer:
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results on combined dataset *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write(f"{key} = {value}\n")
                if any(eval_state):
                    curr_step = eval_state["curr_step"]
                    writer.write(
                        "\n\n Current state (In Compute Metrics Function) \n\n"
                    )
                    for key, value in eval_state.items():
                        writer.write(f"{key} : {value} \n")
            # here we probably want separate predictions for each dataset:
            if training_args.load_best_model_at_end:
                model_path = training_args.output_dir
                if model_name == "cnn" or model_name == "lstm":
                    # non-HF models need manually passed config args
                    model = model_type.from_pretrained(
                        model_path,
                        vocab_size=len(tokenizer),
                        task_names=task_names,
                        num_labels_dict=num_labels,
                    )
                else:
                    model = model_type.from_pretrained(model_path)

                trainer = Trainer(  # make trainer from best model
                    model=model,
                    args=training_args,
                    train_dataset=dataset.processed_dataset.get("train", None),
                    eval_dataset=dataset.processed_dataset.get("validation", None),
                    compute_metrics=build_compute_metrics_fn(
                        task_names, model, dataset
                    ),
                )
                # use trainer to predict

            (task_to_label_packet, _) = (
                current_prediction_packet.pop()
                if len(current_prediction_packet) > 0
                else (None, None)
            )

            for dataset_ind, dataset_path in enumerate(data_args.data_dir):
                subdir = os.path.split(dataset_path.rstrip("/"))[1]
                output_eval_predictions_file = os.path.join(
                    training_args.output_dir,
                    f"eval_predictions_{subdir}_{dataset_ind}_{curr_step}.tsv",
                )

                dataset_dev_segment = get_dataset_segment(
                    "validation", dataset_ind, dataset
                )
                if training_args.error_analysis:
                    out_table = process_prediction(
                        task_names=dataset.tasks,
                        error_analysis=True,
                        output_prob=training_args.output_prob,
                        character_level=data_args.character_level,
                        task_to_label_packet=task_to_label_packet,
                        eval_dataset=dataset_dev_segment,
                        task_to_label_space=task_to_label_space,
                        output_mode=output_mode,
                    )

                    out_table.to_csv(
                        output_eval_predictions_file,
                        sep="\t",
                        index=True,
                        header=True,
                        quoting=csv.QUOTE_NONE,
                        escapechar="\\",
                    )

        eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        trainer.compute_metrics = None
        # FIXME: this part hasn't been updated for the MTL setup so it doesn't work anymore since
        # predictions is generalized to be a list of predictions and the output needs to be different for each kin.
        # maybe it's ok to only handle classification since it has a very straightforward output format and evaluation,
        # while for relations we can punt to the user to just write their own eval code.
        if trainer.is_world_process_zero():
            for dataset_ind, dataset_path in enumerate(data_args.data_dir):
                subdir = os.path.split(dataset_path.rstrip("/"))[1]
                output_test_predictions_file = os.path.join(
                    training_args.output_dir,
                    f"test_predictions_{subdir}_{dataset_ind}.tsv",
                )

                dataset_test_segment = get_dataset_segment("test", dataset_ind, dataset)
                raw_test_predictions = trainer.predict(
                    test_dataset=dataset_test_segment
                )

                (
                    task_to_label_packet,
                    _,
                ) = restructure_prediction(
                    task_names=dataset.tasks,
                    raw_prediction=raw_test_predictions,
                    max_seq_length=data_args.max_seq_length,
                    tagger=tagger,
                    relations=relations,
                    output_prob=training_args.output_prob,
                )

                out_table = process_prediction(
                    task_names=dataset.tasks,
                    error_analysis=False,
                    output_prob=training_args.output_prob,
                    character_level=data_args.character_level,
                    task_to_label_packet=task_to_label_packet,
                    eval_dataset=dataset_test_segment,
                    task_to_label_space=task_to_label_space,
                    output_mode=output_mode,
                )

                out_table.to_csv(
                    output_test_predictions_file,
                    sep="\t",
                    index=True,
                    header=True,
                    quoting=csv.QUOTE_NONE,
                    escapechar="\\",
                )
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

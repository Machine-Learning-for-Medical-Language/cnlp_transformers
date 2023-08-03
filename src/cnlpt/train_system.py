# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on clinical NLP tasks"""
import logging
import os
from os.path import basename, dirname, join, exists
import sys

from typing import Callable, Dict, Optional, List, Union, Any
from filelock import FileLock
import tempfile
import math

from enum import Enum

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel, EvalPrediction
from transformers.training_args import IntervalStrategy
from transformers.data.processors.utils import InputFeatures
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP
from torch.optim import AdamW
from transformers.file_utils import CONFIG_NAME
from huggingface_hub import hf_hub_url

import sys
sys.path.append(os.path.join(os.getcwd()))
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset, DataTrainingArguments
from .cnlp_metrics import cnlp_compute_metrics
from .cnlp_args import CnlpTrainingArguments, ModelArguments
from .cnlp_predict import write_predictions_for_dataset
from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from .BaselineModels import CnnSentenceClassifier, LstmSentenceClassifier
from .HierarchicalTransformer import HierarchicalModel

import requests

from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
)
import json

AutoConfig.register("cnlpt", CnlpConfig)

logger = logging.getLogger(__name__)

def is_hub_model(model_name):
    # check if it's a model on the huggingface model hub:
    try:
        url = hf_hub_url(model_name, CONFIG_NAME)
        r = requests.head(url)
        if r.status_code == 200:
            return True
    except:
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


def encoder_inferred(model_name_or_path: str) -> bool:
    return is_hub_model(model_name_or_path) or not is_cnlpt_model(model_name_or_path)


def main(
    json_file: Optional[str] = None,
    json_obj: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CnlpTrainingArguments))
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: CnlpTrainingArguments

    if json_file is not None and json_obj is not None:
        raise ValueError('cannot specify json_file and json_obj')

    if json_file is not None:
        model_args, data_args, training_args = parser.parse_json_file(json_file=json_file)
    elif json_obj is not None:
        model_args, data_args, training_args = parser.parse_dict(json_obj)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
        (training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16)
    )
    logger.info("Training/evaluation parameters %s" % training_args)
    logger.info("Data parameters %s" % data_args)
    logger.info("Model parameters %s" % model_args)
    # Set seed
    set_seed(training_args.seed)


    # Load tokenizer: Need this first for loading the datasets
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.encoder_name,
        cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        additional_special_tokens=['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
    )

    model_name = model_args.model
    hierarchical = model_name == 'hier'

    # Get datasets
    dataset = (
        ClinicalNlpDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir, hierarchical=hierarchical,)
    )

    try:
        task_names = data_args.task_name
        num_labels = {}
        output_mode = {}
        tagger = {}
        relations = {}
        for task in dataset.tasks_to_labels.keys():
            num_labels[task] = len(dataset.tasks_to_labels[task])
            task_output_mode = dataset.output_modes[task]
            output_mode[task] = task_output_mode
            tagger[task] = (task_output_mode == tagging)
            relations[task] = (task_output_mode == relex)

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_name == 'cnn':
        model = CnnSentenceClassifier(len(tokenizer),
                                      task_names=task_names,
                                      num_labels_dict=num_labels,
                                      embed_dims=model_args.cnn_embed_dim,
                                      num_filters=model_args.cnn_num_filters,
                                      filters=model_args.cnn_filter_sizes,
                                      )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = join(model_args.encoder_name, 'pytorch_model.bin')
        if exists(model_path):
            model.load_state_dict(torch.load(model_path))
    elif model_name == 'lstm':
        model = LstmSentenceClassifier(len(tokenizer),
                                       task_names=task_names,
                                       num_labels_dict=num_labels,
                                       embed_dims=model_args.lstm_embed_dim,
                                       hidden_size=model_args.lstm_hidden_size,
                                       )
        # Check if the caller specified a saved model to load (e.g., for an inference-only run)
        model_path = join(model_args.encoder_name, 'pytorch_model.bin')
        if exists(model_path):
            model.load_state_dict(torch.load(model_path))
    elif model_name == 'hier':
        encoder_name = model_args.config_name if model_args.config_name else model_args.encoder_name
        if encoder_inferred(encoder_name):
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=data_args.task_name if data_args.task_name is not None else dataset.tasks,
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
                )
            )
            # num_tokens=len(tokenizer))
            config.vocab_size = len(tokenizer)

            model = HierarchicalModel(
                config=config,
                class_weights=dataset.class_weights,
                freeze=training_args.freeze,
            )
        else:
            # use a checkpoint from an existing model
            AutoModel.register(CnlpConfig, HierarchicalModel)

            config = AutoConfig.from_pretrained(
                    encoder_name,
                    cache_dir=model_args.cache_dir,
                    layer=model_args.layer
                )
            config.finetuning_task = data_args.task_name
            config.relations = relations
            config.tagger = tagger
            config.label_dictionary = {} # this gets filled in later

            ## TODO: check if user overwrote parameters in command line that could change behavior of the model and warn
            #if data_args.chunk_len is not None:

            logger.info("Loading pre-trained hierarchical model...")
            model = AutoModel.from_pretrained(encoder_name, config=config)

            model.remove_task_classifiers()
            for task in data_args.task_name:
                model.add_task_classifier(task, dataset.get_labels()[task])
            model.set_class_weights(dataset.class_weights)

    else:
        # by default cnlpt model, but need to check which encoder they want
        encoder_name = model_args.encoder_name

        # TODO check when download any pretrained language model to local disk, if
        # the following condition "is_hub_model(encoder_name)" works or not.
        if not encoder_inferred(encoder_name):
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
            AutoModel.register(CnlpConfig, CnlpModelForClassification)

            
            # Load the cnlp configuration using AutoConfig, this will not override
            # the arguments from trained cnlp models. While using CnlpConfig will override
            # the model_type and model_name of the encoder.
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.encoder_name,
                cache_dir=model_args.cache_dir,
            )

            if training_args.do_train:
                # Setting 1) only load weights from the encoder
                raise NotImplementedError('This functionality has not been restored yet')
                model = CnlpModelForClassification(
                        model_path = model_args.encoder_name,
                        config=config,
                        cache_dir=model_args.cache_dir,
                        tagger=tagger,
                        relations=relations,
                        class_weights=dataset.class_weights,
                        final_task_weight=training_args.final_task_weight,
                        use_prior_tasks=model_args.use_prior_tasks,
                        argument_regularization=model_args.arg_reg)
                delattr(model, 'classifiers')
                delattr(model, 'feature_extractors')
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
                    bias_fit=training_args.bias_fit)

        else:
            # This only works when model_args.encoder_name is one of the 
            # model card from https://huggingface.co/models
            # By default, we use model card as the starting point to fine-tune
            encoder_name = model_args.config_name if model_args.config_name else model_args.encoder_name
            config = CnlpConfig(
                encoder_name=encoder_name,
                finetuning_task=data_args.task_name,
                layer=model_args.layer,
                tokens=model_args.token,
                num_rel_attention_heads=model_args.num_rel_feats,
                rel_attention_head_dims=model_args.head_features,
                tagger=tagger,
                relations=relations,
                label_dictionary=dataset.get_labels(),
                #num_tokens=len(tokenizer),
            )
            config.vocab_size = len(tokenizer)
            model = CnlpModelForClassification(
                config=config,
                class_weights=dataset.class_weights,
                final_task_weight=training_args.final_task_weight,
                freeze=training_args.freeze,
                bias_fit=training_args.bias_fit,)

    best_eval_results = None
    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results.txt"
    )

    if training_args.do_train:
        # TODO: This assumes that if there are multiple training sets, they all have the same length, but
        # in the future it would be nice to be able to have multiple heterogeneous datasets
        batches_per_epoch = math.ceil(dataset.num_train_instances / training_args.train_batch_size)
        total_steps = int(training_args.num_train_epochs * batches_per_epoch // training_args.gradient_accumulation_steps)

        if training_args.evals_per_epoch > 0:
            logger.warning('Overwriting the value of logging steps based on provided evals_per_epoch argument')
            # steps per epoch factors in gradient accumulation steps (as compared to batches_per_epoch above which doesn't)
            steps_per_epoch = int(total_steps // training_args.num_train_epochs)
            training_args.eval_steps = steps_per_epoch // training_args.evals_per_epoch
            training_args.evaluation_strategy = IntervalStrategy.STEPS
            # This will save model per epoch
            # training_args.save_strategy = IntervalStrategy.EPOCH
        elif training_args.do_eval:
            logger.info('Evaluation strategy not specified so evaluating every epoch')
            training_args.evaluation_strategy = IntervalStrategy.EPOCH

    def build_compute_metrics_fn(task_names: List[str], model, dataset: ClinicalNlpDataset) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):

            metrics = {}
            task_scores = []
            task_label_ind = 0
            
            for task_ind,task_name in enumerate(task_names):
                if tagger[task_name]:
                    preds = np.argmax(p.predictions[task_ind], axis=2)
                    # labels will be -100 where we don't need to tag
                elif relations[task_name]:
                    preds = np.argmax(p.predictions[task_ind], axis=3)
                else:
                    preds = np.argmax(p.predictions[task_ind], axis=1)

                if relations[task_name]:
                    # relation labels
                    labels = p.label_ids[:,:,task_label_ind:task_label_ind+data_args.max_seq_length].squeeze()
                    task_label_ind += data_args.max_seq_length
                elif p.label_ids.ndim == 3:
                    if tagger[task_name]:
                        labels = p.label_ids[:,:, task_label_ind:task_label_ind+1].squeeze()
                    else:
                        labels = p.label_ids[:, 0, task_label_ind].squeeze()
                    task_label_ind += 1
                elif p.label_ids.ndim == 2:
                    labels = p.label_ids[:,task_ind].squeeze()

                metrics[task_name] = cnlp_compute_metrics(task_name, preds, labels, 
                                                          dataset.output_modes[task_name], 
                                                          dataset.tasks_to_labels[task_name])
                # FIXME - Defaulting to accuracy for model selection score, when it should be task-specific
                task_scores.append( metrics[task_name].get('one_score', np.mean(metrics[task_name].get('f1'))))
                #task_scores.append(processor.get_one_score(metrics.get(task_name, metrics.get(task_name.split('-')[0], None))))

            one_score = sum(task_scores) / len(task_scores)

            if not model is None:
                if not hasattr(model, 'best_score') or one_score > model.best_score:
                    # For convenience, we also re-save the tokenizer to the same directory,
                    # so that you can share your model easily on huggingface.co/models =)
                    if trainer.is_world_process_zero():
                        if training_args.do_train:
                            trainer.save_model()
                            tokenizer.save_pretrained(training_args.output_dir)
                            if model_name == 'cnn' or model_name == 'lstm':
                                with open(os.path.join(training_args.output_dir, 'config.json'), 'w') as f:
                                    json.dump(model_args.to_dict(), f)
                        for task_ind,task_name in enumerate(metrics):
                            with open(output_eval_file, "w") as writer:
                                # logger.info("***** Eval results for task %s *****" % (task_name))
                                for key, value in metrics[task_name].items():
                                    # logger.info("  %s = %s", key, value)
                                    writer.write("%s = %s\n" % (key, value))
                    model.best_score = one_score
                    model.best_eval_results = metrics

            return metrics

        return compute_metrics_fn

    # Initialize our Trainer
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model='one_score'
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.processed_dataset.get('train', None),
        eval_dataset=dataset.processed_dataset.get('validation', None),
        compute_metrics=build_compute_metrics_fn(task_names, model, dataset),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            # resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )

        # if we didn't do any evaluations during training then no model
        # would have ever been saved. we'll save the model here
        if not hasattr(model, 'best_score'):
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_process_zero():
                trainer.save_model()
                tokenizer.save_pretrained(training_args.output_dir)
                if model_name == 'cnn' or model_name == 'lstm':
                    with open(os.path.join(training_args.output_dir, 'config.json'), 'w') as f:
                        json.dump(model_args, f)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_dataset=dataset.processed_dataset['validation']
        # no evaluation was done prior to now, so we need to evaluate
        if not hasattr(model, 'best_eval_results'):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
        else:
            eval_result = model.best_eval_results
        
        # if there is a stored model, restore it so writing outputs uses a good model

        
        trainer.compute_metrics = None
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results on combined dataset *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            # here we probably want separate predictions for each dataset:
            if training_args.load_best_model_at_end:
                model.load_state_dict(torch.load(join(training_args.output_dir, 'pytorch_model.bin')))  # load best model
                trainer = Trainer(  # maake trainer from best model
                    model=model,
                    args=training_args,
                    train_dataset=dataset.processed_dataset.get('train', None),
                    eval_dataset=dataset.processed_dataset.get('validation', None),
                    compute_metrics=build_compute_metrics_fn(task_names, model, dataset),
                ) 
                # use trainer to predict 
            for dataset_ind,dataset_path in enumerate(data_args.data_dir):
                subdir = os.path.split(dataset_path.rstrip('/'))[1]
                output_eval_predictions_file = os.path.join(training_args.output_dir, f'eval_predictions_%s_%d.txt' % (subdir, dataset_ind))
                write_predictions_for_dataset(output_eval_predictions_file, 
                                              trainer,
                                              dataset,
                                              'validation',
                                              dataset_ind,
                                              output_mode,
                                              tokenizer)

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
                subdir = os.path.split(dataset_path.rstrip('/'))[1]
                output_test_predictions_file = os.path.join(training_args.output_dir, f'test_predictions_%s_%d.txt' % (subdir, dataset_ind))
                write_predictions_for_dataset(output_test_predictions_file, 
                                                trainer,
                                                dataset,
                                                'test',
                                                dataset_ind,
                                                output_mode,
                                                tokenizer)

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

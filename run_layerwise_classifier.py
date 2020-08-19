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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
from filelock import FileLock
import time

from enum import Enum

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.data.processors.utils import InputFeatures
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.metrics import acc_and_f1
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features
from EarlyLayerRobertaForSequenceClassification import EarlyLayerRobertaForSequenceClassification

from cnlp_processors import cnlp_processors
# from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # Only allowed task is Negation, don't need this field from Glue
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(cnlp_processors.keys())})
        
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

# class NegationProcessor(DataProcessor):
#     """ Processor for the sdfa shared task negation datasets """
#     def get_example_from_tensor_dict(self, tensor_dict):
#         """See base class."""
#         return InputExample(
#             tensor_dict["idx"].numpy(),
#             tensor_dict["sentence"].numpy().decode("utf-8"),
#             None,
#             str(tensor_dict["label"].numpy()),
#         )

#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

#     def get_labels(self):
#         """See base class."""
#         return ["-1", "1"]

#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training, dev and test sets."""
#         test_mode = set_type == "test"
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             if test_mode:
#                 text_a = line[0]
#                 label = None
#             else:
#                 # flip the signs so that 1 is negated, that way the f1 calculation is automatically
#                 # the f1 score for the negated label.
#                 label = str( -1 * int(line[0]) )
#                 text_a = '\t'.join(line[1:])
#             examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#         return examples
    
class CnlpDataset(Dataset):
    """ Copy-pasted from GlueDataset with glue task-specific code changed
        moved into here to be self-contained
    """
    args: DataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = cnlp_processors[args.task_name]()
        self.output_mode = 'classification'
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length),
            ),
        )
        label_list = self.processor.get_labels()
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    layer: Optional[int] = field(
        default=12, metadata={"help": "Which layer's CLS ('<s>') token to use"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
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
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = 2
        output_mode = 'classification'
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task='negation',
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        additional_special_tokens=('<e>', '</e>')
    )
    model = EarlyLayerRobertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                layer=model_args.layer)
    
    model.resize_token_embeddings(len(tokenizer))
    
    # Get datasets
    train_dataset = (
        CnlpDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        CnlpDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        CnlpDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            return acc_and_f1(preds, p.label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        trainer.compute_metrics = build_compute_metrics_fn()
        eval_result = trainer.evaluate(eval_dataset=eval_dataset)

        output_eval_file = os.path.join(
            training_args.output_dir, f"eval_results.txt"
        )
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        predictions = trainer.predict(test_dataset=test_dataset).predictions
        if output_mode == "classification":
            predictions = np.argmax(predictions, axis=1)

        output_test_file = os.path.join(
            training_args.output_dir, f"test_results.txt"
        )
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for index, item in enumerate(predictions):
                    item = test_dataset.get_labels()[item]
                    writer.write("%s\n" % (item))
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

import os
from os.path import basename, dirname
import time
import logging

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
from filelock import FileLock
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

tasks = {'polarity', 'dtr', 'alink', 'alinkx'}

def cnlp_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "polarity":
        return acc_and_f1(preds, labels)
    elif task_name == "dtr":
        return acc_and_f1(preds, labels)
    elif task_name == "alinkx":
        return acc_and_f1(preds, labels)

class NegationProcessor(DataProcessor):
    """ Processor for the negation datasets """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["-1", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if test_mode:
                # Some test sets have labels and some do not. discard the label if it has it but hvae to check so
                # we know which part of the line has the data.
                if line[0] in self.get_labels():
                    text_a = '\t'.join(line[1:])
                else:
                    text_a = '\t'.join(line[0:])
                label = None
            else:
                # flip the signs so that 1 is negated, that way the f1 calculation is automatically
                # the f1 score for the negated label.
                label = str( -1 * int(line[0]) )
                text_a = '\t'.join(line[1:])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class DtrProcessor(DataProcessor):
    """ Processor for DocTimeRel datasets """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["BEFORE", "OVERLAP", "BEFORE/OVERLAP", "AFTER"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if test_mode:
                if line[0] in self.get_labels():
                    text_a = '\t'.join(line[1:])
                else:
                    text_a = '\t'.join(line[0:])
                label = None
            else:
                label = line[0]
                text_a = '\t'.join(line[1:])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class AlinkxProcessor(DataProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status."""
    def get_train_examples(self, data_dir, ignore_labels=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", ignore_labels)

    def get_dev_examples(self, data_dir, ignore_labels=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", ignore_labels)

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["None", "CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        return np.mean(results['f1'][1:])
    
    def _create_examples(self, lines, set_type, ignore_labels=False):
        """Creates examples for the training and dev sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if ignore_labels:
                label = IGNORE_LABEL
                
            if test_mode:
                if line[0] in self.get_labels():
                    text_a = '\t'.join(line[1:])
                else:
                    text_a = '\t'.join(line[0:])
                label = None
            else:
                label = line[0]
                text_a = '\t'.join(line[1:])
                
#             if set_type=='train' and label in self.downsampling:
#                 dart = random.random()
#                 # if downsampling is set to 0.1 that downsample that class to 10%.
#                 # so if our randomly generated number is bigger than our downsampling rate
#                 # we skip this instance.
#                 if dart > self.downsampling[label]:
#                     continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
cnlp_processors = {'polarity': NegationProcessor,
                   'dtr': DtrProcessor,
#                    'alink': AlinkProcessor,
                   'alinkx': AlinkxProcessor
                  }

cnlp_num_labels = { 'polarity': 2,
                    'dtr': 4,
                    'alink': 4,
                    'alinkx': 5
                  }
                  
cnlp_output_modes = {'polarity': 'classification',
                'dtr': 'classification',
                'alink': 'classification',
                'alinkx': 'classification'}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

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
    
class ClinicalNlpDataset(Dataset):
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
        self.output_mode = cnlp_output_modes[args.task_name]

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        dataset = basename(dirname(args.data_dir)) if args.data_dir[-1] == '/' else basename(args.data_dir)
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                args.task_name, dataset, mode.value, tokenizer.__class__.__name__, str(args.max_seq_length),
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


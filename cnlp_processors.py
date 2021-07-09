import os
from os.path import basename, dirname
import time
import logging

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union
from transformers.data.processors.utils import DataProcessor, InputExample
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score
import numpy as np
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_cls

logger = logging.getLogger(__name__)

def tagging_metrics(task_name, preds, labels):
    processor = cnlp_processors[task_name]()
    label_set = processor.get_labels()

    preds = preds.flatten()
    labels = labels.flatten().astype('int')

    pred_inds = np.where(labels != -100)
    preds = preds[pred_inds]
    labels = labels[pred_inds]

    pred_seq = [ label_set[x] for x in preds]
    label_seq = [ label_set[x] for x in labels]

    num_correct = (preds==labels).sum()

    acc = num_correct / len(preds)
    f1 = f1_score(preds, labels, average=None)

    return {'acc': acc, 'token_f1': f1, 'f1': seq_f1([pred_seq], [label_seq]), 'report':'\n'+seq_cls([pred_seq], [label_seq])}

def relation_metrics(task_name, preds, labels):

    processor = cnlp_processors[task_name]()
    label_set = processor.get_labels()

    # If we are using the attention-based relation extractor, many impossible pairs
    # are set to -100 so pytorch loss functions ignore them. We need to make sure the
    # scorer also ignores them.
    relevant_inds = np.where(labels != -100)
    relevant_labels = labels[relevant_inds]
    relevant_preds = preds[relevant_inds]

    num_correct = (relevant_labels == relevant_preds).sum()
    acc = num_correct / len(relevant_preds)

    recall = recall_score(relevant_preds, relevant_labels)
    precision = precision_score(relevant_preds, relevant_labels)
    f1_report = f1_score(relevant_preds, relevant_labels, average=None)

    return {'f1': f1_report, 'acc': acc, 'recall':recall, 'precision':precision }

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

tasks = {'polarity', 'dtr', 'alink', 'alinkx', 'tlink'}

def cnlp_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "polarity":
        return acc_and_f1(preds, labels)
    elif task_name == "dtr":
        return acc_and_f1(preds, labels)
    elif task_name == "alink":
        return acc_and_f1(preds, labels)
    elif task_name == "alinkx":
        return acc_and_f1(preds, labels)
    elif task_name == 'tlink':
        return acc_and_f1(preds, labels)
    elif task_name == 'conmod':
        return acc_and_f1(preds, labels)
    elif task_name == 'timecat':
        return acc_and_f1(preds, labels)
    elif task_name == 'timex' or task_name == 'event':
        return tagging_metrics(task_name, preds, labels)
    elif task_name == 'tlink-sent':
        return relation_metrics(task_name, preds, labels)

class CnlpProcessor(DataProcessor):
    def __init__(self, downsampling={}):
        super().__init__()
        self.downsampling = downsampling

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

    def _create_examples(self, lines, set_type, sequence=False, relations=False):
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
                if sequence:
                    label = line[0].split(' ')
                elif relations:
                    if line[0] == 'None':
                        label = []
                    else:
                        label = [x[1:-1].split(',') for x in line[0].split(' , ')]
                else:
                    label = line[0]
                text_a = '\t'.join(line[1:])

            if set_type=='train' and not sequence and not relations and label in self.downsampling:
                dart = random.random()
                # if downsampling is set to 0.1 then sample 10% of those instances.
                # so if our randomly generated number is bigger than our downsampling rate
                # we skip this instance.
                if dart > self.downsampling[label]:
                    continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class LabeledSentenceProcessor(CnlpProcessor):
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, sequence=False)

class NegationProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        """See base class."""
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]


class DtrProcessor(LabeledSentenceProcessor):
    """ Processor for DocTimeRel datasets """
    def get_labels(self):
        """See base class."""
        return ["BEFORE", "OVERLAP", "BEFORE/OVERLAP", "AFTER"]

    def get_one_score(self, results):
        return np.mean(results['acc'])

class AlinkxProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status."""

    def get_labels(self):
        """See base class."""
        return ["None", "CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        return np.mean(results['f1'][1:])

class AlinkProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status."""
    def get_labels(self):
        """See base class."""
        return ["CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        return np.mean(results['f1'])

class ContainsProcessor(LabeledSentenceProcessor):
    """ Processor for narrative container relation (THYME). Describes the contains relation status between the 
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2, 
    CONTAINS-1 - arg 2 contains arg 1"""
    def get_labels(self):
        """See base class."""
        return ["NONE", "CONTAINS", "CONTAINS-1"]

class TlinkProcessor(LabeledSentenceProcessor):
    """ Processor for narrative container relation (THYME). Describes the contains relation status between the 
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2, 
    CONTAINS-1 - arg 2 contains arg 1"""
    def get_labels(self):
        """See base class."""
        return ["BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP" ]

    def get_one_score(self, results):
        return np.mean(results['f1'])
    
class TimeCatProcessor(LabeledSentenceProcessor):
    """Processor for an THYME time expression dataset
    The classifier version of the task is _given_ a time class, label its time category (see labels below)."""
    def get_labels(self):
        """See base class."""
        return ["DATE", "DOCTIME", "DURATION", "PREPOSTEXP", "QUANTIFIER", "SECTIONTIME", "SET", "TIME"]

    def get_one_score(self, results):
        return results['acc']

class ContextualModalityProcessor(LabeledSentenceProcessor):
    """Processor for a contexutal modality dataset """
    def get_labels(self):
        """See base class."""
        return ["ACTUAL", "HYPOTHETICAL", "HEDGED", "GENERIC"]

    def get_one_score(self, results):
        # actual is the default and it's very common so we use the macro f1 of non-default categories for model selection.
        return np.mean(results['f1'][1:])

class RelationProcessor(CnlpProcessor):
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, relations=True)

class TlinkRelationProcessor(RelationProcessor):
    def get_one_score(self, results):
        # the 0th category is None
        return np.mean(results['f1'][1:])
    
    def get_labels(self):
        return ['None', 'CONTAINS']
        # return ['None', 'CONTAINS', 'OVERLAP', 'BEFORE', 'BEGINS-ON', 'ENDS-ON']

class SequenceProcessor(CnlpProcessor):
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, sequence=True)

class TimexProcessor(SequenceProcessor):
    def get_one_score(self, results):
        return results['f1']

    def get_labels(self):
        return ["O", "B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME",
                "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME"
                ]

class EventProcessor(SequenceProcessor):
    def get_one_score(self, results):
        return results['f1']
    
    def get_labels(self):
        return ["O", "B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
                ,"I-BEFORE/OVERLAP","I-OVERLAP"]
        # return ['B-EVENT', 'I-EVENT', 'O']

cnlp_processors = {'polarity': NegationProcessor,
                   'dtr': DtrProcessor,
                   'alink': AlinkProcessor,
                   'alinkx': AlinkxProcessor,
                   'tlink': TlinkProcessor,
                   'nc': ContainsProcessor,
                   'timecat': TimeCatProcessor,
                   'conmod': ContextualModalityProcessor,
                   'timex': TimexProcessor,
                   'event': EventProcessor,
                   'tlink-sent': TlinkRelationProcessor,
                  }

# cnlp_num_labels = { 'polarity': 2,
#                     'dtr': 4,
#                     'alink': 4,
#                     'alinkx': 5,
#                     'nc': 3,
#                     'tlink': 5,
#                     'timecat': 8,
#                     'conmod': 4,
#                     'timex': 17,
#                     'event': 9,
#                   }

classification = 'classification'
tagging = 'tagging'
relex = 'relations'

cnlp_output_modes = {'polarity': classification,
                'dtr': classification,
                'alink': classification,
                'alinkx': classification,
                'tlink': classification,
                'nc': classification,
                'timecat': classification,
                'conmod': classification,
                'timex': tagging,
                'event': tagging,
                'tlink-sent': relex
                }


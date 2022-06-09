import os
import random
from abc import ABC, abstractmethod
import logging
import json

from typing import List
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.data.metrics import simple_accuracy
from sklearn.metrics import f1_score, recall_score, precision_score
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

    pred_seq = [label_set[x] for x in preds]
    label_seq = [label_set[x] for x in labels]

    num_correct = (preds == labels).sum()

    acc = num_correct / len(preds)
    f1 = f1_score(labels, preds, average=None)

    return {'acc': acc, 'token_f1': fix_np_types(f1), 'f1': fix_np_types(seq_f1([label_seq], [pred_seq])), 'report':'\n'+seq_cls([label_seq], [pred_seq])}

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

    recall = recall_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    precision = precision_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    f1_report = f1_score(y_true=relevant_labels, y_pred=relevant_preds, average=None)

    return {'f1': fix_np_types(f1_report), 'acc': acc, 'recall':fix_np_types(recall), 'precision':fix_np_types(precision) }

def fix_np_types(input_variable):
    ''' in the mtl classification setting, f1 is an array, and when the HF library
        tries to write out the trainig history to a json file it will throw an error.
        Here, we just check whether it's an numpy array and if so convert to a list.
    '''
    if isinstance(input_variable, np.ndarray):
        return list(input_variable)
    
    return input_variable

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    recall = recall_score(y_true=labels, y_pred=preds, average=None)
    precision = precision_score(y_true=labels, y_pred=preds, average=None)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)
    
    return {
        "acc": fix_np_types(acc),
        "f1": fix_np_types(f1),
        "acc_and_f1": fix_np_types((acc + f1) / 2),
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision)
    }

tasks = {'polarity', 'dtr', 'alink', 'alinkx', 'tlink'}

def cnlp_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "polarity" or task_name == "uncertainty" or task_name == "history" or task_name == "subject":
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
    elif task_name.startswith('i2b22008') or task_name.startswith('mimic_radi'):
        return { 'f1': fix_np_types(f1_score(y_true=labels, y_pred=preds, average=None))} #acc_and_f1(preds, labels)
    elif task_name == 'timex' or task_name == 'event' or task_name == 'dphe':
        return tagging_metrics(task_name, preds, labels)
    elif task_name == 'tlink-sent':
        return relation_metrics(task_name, preds, labels)
    elif cnlp_output_modes[task_name] == classification:
        logger.warn("Choosing accuracy and f1 as default metrics; modify cnlp_compute_metrics() to customize for this task.")
        return acc_and_f1(preds, labels)
    else:
        raise Exception('There is no metric defined for this task in function cnlp_compute_metrics()')


class CnlpBaseProcessor(DataProcessor, ABC):
    @property
    @abstractmethod
    def relations(self) -> bool:
        pass

    @property
    @abstractmethod
    def sequence(self) -> bool:
        pass

    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def get_one_score(self, results):
        pass

    @abstractmethod
    def get_num_tasks(self):
        pass


class CnlpProcessor(CnlpBaseProcessor, ABC):
    """
    Abstract base class for single-task processors
    """
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

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if test_mode:
                # Some test sets have labels and some do not. discard the label if it has it but have to check so
                # we know which part of the line has the data.
                if line[0] in self.get_labels():
                    text_a = '\t'.join(line[1:])
                else:
                    text_a = '\t'.join(line[0:])
                label = None
            else:
                if self.sequence:
                    label = line[0].split(' ')
                elif self.relations:
                    if line[0].lower() == 'none':
                        label = []
                    else:
                        label = [x[1:-1].split(',') for x in line[0].split(' , ')]
                else:
                    label = line[0]
                text_a = '\t'.join(line[1:])

            if set_type=='train' and not self.sequence and not self.relations and label in self.downsampling:
                dart = random.random()
                # if downsampling is set to 0.1 then sample 10% of those instances.
                # so if our randomly generated number is bigger than our downsampling rate
                # we skip this instance.
                if dart > self.downsampling[label]:
                    continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_num_tasks(self):
        return 1


###############################
# Labeled sentence processors #
###############################
class LabeledSentenceProcessor(CnlpProcessor, ABC):
    """
    Abstract base class for labeled sentence processors
    """
    sequence = False
    relations = False

    def get_one_score(self, results):
        return results['f1'].mean()

class NegationProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        """See base class."""
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]

class UncertaintyProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        """See base class."""
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]

class HistoryProcessor(LabeledSentenceProcessor):
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

class UciDrugSentimentProcessor(LabeledSentenceProcessor):
    def get_labels(self):
        return ['Low', 'Medium', 'High']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class Mimic_7_Processor(LabeledSentenceProcessor):
    def get_labels(self):
        return ['7+', '7-']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class Mimic_3_Processor(LabeledSentenceProcessor):
    def get_labels(self):
        return ['3+', '3-']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class CovidProcessor(LabeledSentenceProcessor):
    def get_labels(self):
        return ['negative', 'positive']
    
    def get_one_score(self, results):
        return results['f1'][1]


#######################
# Relation processors #
#######################
class RelationProcessor(CnlpProcessor, ABC):
    """
    Abstract base class for relation processors
    """
    sequence = False
    relations = True


class TlinkRelationProcessor(RelationProcessor):
    def get_one_score(self, results):
        # the 0th category is None
        return np.mean(results['f1'][1:])
    
    def get_labels(self):
        return ['None', 'CONTAINS']
        #return ['None', 'CONTAINS', 'NOTED-ON']
        # return ['None', 'CONTAINS', 'OVERLAP', 'BEFORE', 'BEGINS-ON', 'ENDS-ON']


#######################
# Sequence processors #
#######################
class SequenceProcessor(CnlpProcessor, ABC):
    """
    Abstract base class for sequence processors
    """
    sequence = True
    relations = False


class TimexProcessor(SequenceProcessor):
    def get_one_score(self, results):
        return results['f1']

    def get_labels(self):
        return ["O", "B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME","B-SECTIONTIME","B-DOCTIME",
                "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME","I-SECTIONTIME","I-DOCTIME",
                ]

class EventProcessor(SequenceProcessor):
    def get_one_score(self, results):
        return results['f1']
    
    def get_labels(self):
        return ["O", "B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
                ,"I-BEFORE/OVERLAP","I-OVERLAP"]
        # return ['B-EVENT', 'I-EVENT', 'O']

class DpheProcessor(SequenceProcessor):
    def get_one_score(self, results):
        return results['f1']

    def get_labels(self):
        return ["O", "B-drug","B-dosage","B-duration","B-frequency","B-form","B-route","B-strength",
                "I-drug","I-dosage","I-duration","I-frequency","I-form","I-route","I-strength"
                ]


##################################
# Multi-task learning processors #
##################################
class MTLClassifierProcessor(CnlpBaseProcessor, ABC):
    """
    Abstract base class for multi-task learning classifier processors
    """
    sequence = False
    relations = False

    subset: set

    def __init__(self, subset=None):
        self.subset = set(subset) if subset is not None else set()

    def get_example_from_tensor_dict(self, tensor_dict):
        raise RuntimeError("get_example_from_tensor_dict not permitted for MTL tasks")

    def get_classifiers(self) -> List[str]:
        pass

    def _get_classifiers(self, classifiers) -> List[str]:
        if self.subset:
            bad_clfs = self.subset - set(classifiers)
            if bad_clfs:
                logger.warning(f"Supplied {', '.join(bad_clfs)} which "
                               f"{'is' if len(bad_clfs) == 1 else 'are'} "
                               f"not in the list of classifiers")
            subset_classifiers = [clf for clf in classifiers if clf in self.subset]
            return subset_classifiers
        return classifiers

    def get_default_label(self) -> str:
        return NotImplemented

    def get_classifier_id(self) -> str:
        return NotImplemented

    def get_train_examples(self, data_dir):
        return self.get_json_examples(os.path.join(data_dir, 'training.json'), 'train')

    def get_dev_examples(self, data_dir):
        return self.get_json_examples(os.path.join(data_dir, 'dev.json'), 'dev')

    def get_test_examples(self, data_dir):
        return self.get_json_examples(os.path.join(data_dir, 'test.json'), 'test')

    def get_json_examples(self, fn, set_type):
        test_mode = set_type == "test"
        examples = []

        with open(fn, 'rt') as f:
            data = json.load(f)
        
        for inst_id, instance in data.items():
            guid = '%s-%s' % (self.get_classifier_id(), inst_id)
            text_a = instance['text']
            label_dict = instance['labels']
            labels = [label_dict.get(x, self.get_default_label()) for x in self.get_classifiers()]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=labels))
        
        return examples

class MimicRadiProcessor(MTLClassifierProcessor):
    def get_classifiers(self):
        classifiers = ["3-", "3+", "7-", "7+"]
        # "3-": "Y", "3+": "N", "7-": "Y", "7+": "N"
        return self._get_classifiers(classifiers)

    def get_labels(self):
        # return [ ["Y", "N"] for x in range(len(self.get_classifiers()))]
        return ["Y", "N"]
    
    def get_num_tasks(self):
        return len(self.get_classifiers())
    
    def get_one_score(self, results):
        print(results)
        #return results #['f1'].mean()
        return np.mean(results['f1'])

    def get_classifier_id(self):
        return 'mimic_radi'

    def get_default_label(self):
        return 'Unlabeled'

class i2b22008Processor(MTLClassifierProcessor):
    """
    Processor for the i2b2-2008 disease classification dataset
    """

    def get_classifiers(self):
        classifiers = [
            'Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes',
            'Gallstones', 'GERD', 'Gout', 'Hypertension',
            'Hypertriglyceridemia', 'Hypercholesterolemia',
            'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency'
        ]
        return self._get_classifiers(classifiers)

    def get_labels(self):
        # return [ ["Unlabeled", "Y", "N", "Q", "U"] for x in range(len(self.get_classifiers()))]
        return ["Unlabeled", "Y", "N", "Q", "U"]

    def get_default_label(self):
        return 'Unlabeled'
    
    def get_classifier_id(self):
        return 'i2b2-2008'

    def get_num_tasks(self):
        return len(self.get_classifiers())
    
    def get_one_score(self, results):
        return np.mean(results['f1'][1:2])

cnlp_processors = {'polarity': NegationProcessor,
                   'uncertainty': UncertaintyProcessor,
                   'history': HistoryProcessor,
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
                   'dphe': DpheProcessor,
                   'i2b22008': i2b22008Processor,
                   'ucidrug': UciDrugSentimentProcessor,
                   'mimic_radi': MimicRadiProcessor,
                   'mimic_3': Mimic_3_Processor,
                   'mimic_7': Mimic_7_Processor,
                   'covid': CovidProcessor
                  }

mtl = 'mtl'
classification = 'classification'
tagging = 'tagging'
relex = 'relations'

cnlp_output_modes = {'polarity': classification,
                'uncertainty': classification,
                'history': classification,
                'dtr': classification,
                'alink': classification,
                'alinkx': classification,
                'tlink': classification,
                'nc': classification,
                'timecat': classification,
                'conmod': classification,
                'timex': tagging,
                'event': tagging,
                'dphe': tagging,
                'tlink-sent': relex,
                'i2b22008': mtl,
                'ucidrug': classification,
                'mimic_radi': mtl,
                'mimic_3': classification,
                'mimic_7': classification,
                'covid': classification
                }


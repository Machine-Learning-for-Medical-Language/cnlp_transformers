"""
Module containing processor classes, evaluation metrics, and output
modes for tasks defined in the library.

Add custom classes here to add new tasks to the library with the following steps:

#. Create a unique ``task_name`` for your task.
#. :data:`cnlp_output_modes` -- Add a mapping from a task name to a
   task type. Currently supported task types are sentence classification,
   tagging, relation extraction, and multi-task sentence classification.
#. Processor class -- Create a subclass of :class:`transformers.DataProcessor`
   for your data source. There are multiple examples to base off of,
   including intermediate abstractions like :class:`LabeledSentenceProcessor`,
   :class:`RelationProcessor`, :class:`SequenceProcessor`, that simplify
   the implementation.
#. :data:`cnlp_processors` -- Add a mapping from your task name to the
   "processor" class you created in the last step.
#. (Optional) -- Modify :func:`cnlp_compute_metrics` to add
   you task. If your task is classification a reasonable default will
   be used so this step would be optional.

.. data:: cnlp_processors

    Mapping from task names to processor classes

    :type: typing.Dict[str, transformers.DataProcessor]

.. data:: cnlp_output_modes

    Mapping from task names to output modes

    :type: typing.Dict[str, str]

"""
import os
import random
from os.path import basename, dirname, join
import time
import logging
import json

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Union, Any
from transformers.data.processors.utils import DataProcessor, InputExample
from datasets import load_dataset
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import numpy  # for Sphinx

logger = logging.getLogger(__name__)

mtl = 'mtl'
classification = 'classification'
tagging = 'tagging'
relex = 'relations'

class CnlpProcessor(DataProcessor):
    """
    Base class for single-task dataset processors

    :param typing.Optional[typing.Dict[str, float]] downsampling: downsampling values for class balance
    """
    def __init__(self, downsampling=None):
        super().__init__()
        if downsampling is None:
            downsampling = {}
        self.downsampling = downsampling

    def get_one_score(self, results):
        """
        Return a single value to use as the score for
        selecting the best model epoch after training.

        :param typing.Dict[str, typing.Any] results: the dictionary of evaluation
            metrics for the current epoch
        :return: a single value; it needs to be of a type that can be
            ordered (preferably, but not necessarily, a float).
        """
        raise NotImplementedError()

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_output_mode(self):
        """
        Return a string indicating what kind of task this is -- e.g., classification, tagging, mtl, relex
        :return: a single string
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type, sequence=False, relations=False):
        """
        **This is an internal function, but it is included in the documentation
        to illustrate the input format for single-task datasets.**

        ----

        Creates examples for the training, dev and test sets from a
        headingless TSV file with one of the following structures:

        * For sequence classification::

            label\ttext

        * For sequence tagging::

            tag1 tag2 ... tagN\ttext

        * For relation tagging::

            <source1,target1> , <source2,target2> , ... , <sourceN,targetN>\ttext

        TODO: check that these formats are correct

        :meta public:
        """
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
                if sequence:
                    label = line[0].split(' ')
                elif relations:
                    if line[0].lower() == 'none':
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

    def get_num_tasks(self):
        return 1

class LabeledSentenceProcessor(CnlpProcessor):
    """
    Base class for labeled sentence dataset processors
    """
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, sequence=False)

    def get_one_score(self, results):
        return results['f1'].mean()

    def get_output_mode(self):
        return classification

class NegationProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]

class UncertaintyProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]

class HistoryProcessor(LabeledSentenceProcessor):
    """ Processor for the negation datasets """
    def get_labels(self):
        return ["-1", "1"]

    def get_one_score(self, results):
        return results['f1'][1]

class DtrProcessor(LabeledSentenceProcessor):
    """ Processor for DocTimeRel datasets """
    def get_labels(self):
        return ["BEFORE", "OVERLAP", "BEFORE/OVERLAP", "AFTER"]

    def get_one_score(self, results):
        return np.mean(results['acc'])

class AlinkxProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status."""

    def get_labels(self):
        return ["None", "CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        return np.mean(results['f1'][1:])

class AlinkProcessor(LabeledSentenceProcessor):
    """Processor for an THYME ALINK dataset (links that describe change in temporal status of an event)
    The classifier version of the task is _given_ an event known to have some aspectual status, label that status."""
    def get_labels(self):
        return ["CONTINUES", "INITIATES", "REINITIATES", "TERMINATES"]

    def get_one_score(self, results):
        return np.mean(results['f1'])

class ContainsProcessor(LabeledSentenceProcessor):
    """ Processor for narrative container relation (THYME). Describes the contains relation status between the 
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2, 
    CONTAINS-1 - arg 2 contains arg 1"""
    def get_labels(self):
        return ["NONE", "CONTAINS", "CONTAINS-1"]

class TlinkProcessor(LabeledSentenceProcessor):
    """ Processor for narrative container relation (THYME). Describes the contains relation status between the 
    two highlighted temporal entities (event or timex). NONE - no relation, CONTAINS - arg 1 contains arg2, 
    CONTAINS-1 - arg 2 contains arg 1"""
    def get_labels(self):
        return ["BEFORE", "BEGINS-ON", "CONTAINS", "ENDS-ON", "OVERLAP" ]

    def get_one_score(self, results):
        return np.mean(results['f1'])

class TimeCatProcessor(LabeledSentenceProcessor):
    """Processor for a THYME time expression dataset
    The classifier version of the task is _given_ a time class, label its time category (see labels below)."""
    def get_labels(self):
        return ["DATE", "DOCTIME", "DURATION", "PREPOSTEXP", "QUANTIFIER", "SECTIONTIME", "SET", "TIME"]

    def get_one_score(self, results):
        return results['acc']

class ContextualModalityProcessor(LabeledSentenceProcessor):
    """Processor for a contextual modality dataset """
    def get_labels(self):
        return ["ACTUAL", "HYPOTHETICAL", "HEDGED", "GENERIC"]

    def get_one_score(self, results):
        # actual is the default and it's very common so we use the macro f1 of non-default categories for model selection.
        return np.mean(results['f1'][1:])

class UciDrugSentimentProcessor(LabeledSentenceProcessor):
    """Processor for the UCI Drug Review sentiment classification dataset"""
    def get_labels(self):
        return ['Low', 'Medium', 'High']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class Mimic_7_Processor(LabeledSentenceProcessor):
    """TODO: docstring"""
    def get_labels(self):
        return ['7+', '7-']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class Mimic_3_Processor(LabeledSentenceProcessor):
    """TODO: docstring"""
    def get_labels(self):
        return ['3+', '3-']

    def get_one_score(self, results):
        return np.mean(results['f1'])

class CovidProcessor(LabeledSentenceProcessor):
    """TODO: docstring"""
    def get_labels(self):
        return ['negative', 'positive']
    
    def get_one_score(self, results):
        return results['f1'][1]

class RelationProcessor(CnlpProcessor):
    """
    Base class for relation extraction dataset processors
    """
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, relations=True)

    def get_output_mode(self):
        return relex

class Thyme1ContainsRelationProcessor(RelationProcessor):
    """TODO: docstring"""
    def get_one_score(self, results):
        # the 0th category is None
        return np.mean(results['f1'][1:])
    
    def get_labels(self):
        return ['None', 'CONTAINS']

class Thyme1AllRelationProcessor(RelationProcessor):
    """TODO: docstring"""
    def get_one_score(self, results):
        # the 0th category is None
        return np.mean(results['f1'][1:])
    def get_labels(self):
        return ['None', 'CONTAINS', 'OVERLAP', 'BEFORE', 'BEGINS-ON', 'ENDS-ON']

class SequenceProcessor(CnlpProcessor):
    """
    Base class for sequence tagging dataset processors
    """
    def _create_examples(self, lines, set_type):
        return super()._create_examples(lines, set_type, sequence=True)

    def get_output_mode(self):
        return tagging

class TimexProcessor(SequenceProcessor):
    """TODO: docstring"""
    def get_one_score(self, results):
        return results['f1']

    def get_labels(self):
        return ["O", "B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME","B-SECTIONTIME","B-DOCTIME",
                "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME","I-SECTIONTIME","I-DOCTIME",
                ]

class EventProcessor(SequenceProcessor):
    """TODO: docstring"""
    def get_one_score(self, results):
        return results['f1']
    
    def get_labels(self):
        return ["O", "B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
                ,"I-BEFORE/OVERLAP","I-OVERLAP"]
        # return ['B-EVENT', 'I-EVENT', 'O']

class DpheProcessor(SequenceProcessor):
    """TODO: docstring"""
    def get_one_score(self, results):
        return results['f1']

    def get_labels(self):
        return ["O", "B-drug","B-dosage","B-duration","B-frequency","B-form","B-route","B-strength",
                "I-drug","I-dosage","I-duration","I-frequency","I-form","I-route","I-strength"
                ]

class MTLClassifierProcessor(DataProcessor):
    """
    Base class for multi-task learning classification dataset processors. 
    This class can be used in the specific multi-task setting where there are multiple tasks with the same
    labels for each data instance. For the more general case of just wanting to have the model
    do multiple different tasks with different datasets and different label sets, they can all
    just be given as separate tasks/data directories at the command line.
    """

    def get_classifiers(self):
        """
        Get the list of classification subtasks in this multi-task setting

        :rtype: typing.List[str]
        :return: a list of task names
        """
        return NotImplemented

    def get_num_tasks(self):
        """
        Get the number of subtasks in this multi-task setting.

        Equivalent to :obj:`len(self.get_classifiers())`.

        :rtype: int
        :return: the number of subtasks
        """
        return len(self.get_classifiers())

    def get_classifier_id(self):
        """
        Get the classifier ID name used in building the GUIDs for the
        :class:`transformers.InputExample` instances.

        Not necessarily equal to the ``task_name`` used as keys for
        :data:`cnlp_processors` and :data:`cnlp_output_modes`.

        :rtype: str
        :return: the value of the classifier ID
        """
        pass

    def get_default_label(self):
        """
        Get the default label to assign to unlabeled instances in the dataset.

        :rtype: str
        :return: the value of the default label
        """
        pass

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Not used.
        """
        return RuntimeError("not implemented for MTL tasks")

    def get_output_mode(self):
        return mtl

    def get_train_examples(self, data_dir):
        return self._get_json_examples(os.path.join(data_dir, 'training.json'), 'train')

    def get_dev_examples(self, data_dir):
        return self._get_json_examples(os.path.join(data_dir, 'dev.json'), 'dev')

    def get_test_examples(self, data_dir):
        return self._get_json_examples(os.path.join(data_dir, 'test.json'), 'test')

    def _get_json_examples(self, fn, set_type):
        """
        **This is an internal function, but it is included in the documentation
        to illustrate the input format for MTL datasets.**

        ----

        Creates examples for the training, dev and test sets
        from a JSON file with the following structure::

            {
                "<guid_1>": {
                    "text": "<text>",
                    "labels: {
                        "<task_1>": "<label>",
                        ...
                    }
                },
                ...
            }

        :param str fn: the path to the dataset file to load
        :param str set_type: the type of split the file contains (e.g. train, dev, test)
        :rtype: typing.List[transformers.InputExample]
        :return: the examples loaded from the file
        :meta public:
        """
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
    """TODO: docstring"""
    def get_classifiers(self):
        return ["3-", "7-"]

    def get_labels(self):
        return ["Y", "N"]
    
    def get_one_score(self, results):
        print(results)
        return np.mean(results['f1'])

    def get_classifier_id(self):
        return 'mimic_radi'

    def get_default_label(self):
        return 'N'

class i2b22008Processor(MTLClassifierProcessor):
    """
    Processor for the i2b2-2008 disease classification dataset
    """
    def get_classifiers(self):
        return ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'Hypertension',
                'Hypertriglyceridemia', 'Hypercholesterolemia', 'OA', 'Obesity', 'OSA', 'PVD', 'Venous Insufficiency']

    def get_labels(self):
        # return [ ["Unlabeled", "Y", "N", "Q", "U"] for x in range(len(self.get_classifiers()))]
        return ["Unlabeled", "Y", "N", "Q", "U"]

    def get_default_label(self):
        return 'Unlabeled'
    
    def get_classifier_id(self):
        return 'i2b2-2008'
    
    def get_one_score(self, results):
        return np.mean(results['f1'][1:2])

class PsychDomainProcessor(MTLClassifierProcessor):
    """
    Processor for the McLean Hospital readmission risk factor domain dataset. Each sentence has a binary label for 7 different readmission risk factors representing whether that risk factor is present in the sentence.
    """
    def get_classifiers(self):
        return ["Appearance", "Content (Thought Content)", "Interpersonal", "Mood", "Occupation", "Process (Thought Process)", "Substance"]
    
    def get_labels(self):
        return ["No", "Yes"]
    
    def get_default_label(self):
        return "No"
    
    def get_classifier_id(self):
        return 'psych_domain'

    def get_one_score(self, results):
        return results['f1'][1]

class InferringProcessor(DataProcessor):
    """
    A special type of processor that tries to infer the details about the dataset from the
    artifacts that are present in the data directory.
    """
    def __init__(self, data_dir):
        super().__init__()

        train_file = dev_file = test_file = None
        data_files = {}
        for fn in os.listdir(data_dir):
            if fn.startswith('train'):
                train_file = fn
                data_files['train'] = join(data_dir, train_file)
            elif fn.startswith('dev') or fn.startswith('valid'):
                dev_file = fn
                data_files['validation'] = join(data_dir, dev_file)
            elif fn.startswith('test'):
                test_file = fn
                data_files['test'] = join(data_dir, test_file)
        
        if (train_file is None and 
            dev_file is None and 
            test_file is None):
            raise ValueError("This dataset doesn't have train, dev, or test files")

        metadata = None
        if train_file is not None:
            ext_check_file = train_file
        elif dev_file is not None:
            ext_check_file = dev_file
        else:
            ext_check_file = test_file

        if ext_check_file.endswith('csv'):
            file_type = 'csv'
        elif ext_check_file.endswith('json'):
            file_type = 'json'
        else:
            raise ValueError('Data file %s has an extension that we cannot handle (tried csv and json)' % (train_file))

        if not train_file is None:
            self.dataset = load_dataset(file_type, data_files=data_files, field='data')

        if file_type == 'json':
            with open(join(data_dir, ext_check_file), 'rt', encoding="utf-8") as f:
                json_file = json.load(f)
                metadata = json_file['metadata']
        elif file_type == 'csv':
            # TODO - get the metadata (task names) from the CSV header (column 0)
            metadata = {}

        self.dataset.metadata = metadata

        any_split = next(iter(self.dataset.values()))
        unique_labels = list(set( any_split[self.dataset.metadata['tasks'][0]]) )
        unique_labels.sort()
        self.labels = unique_labels
        self.classifiers = metadata['tasks']

        print("Loaded dataset has length %d" % (len(self.dataset)))

    def get_train_examples(self, data_dir):
        return self.dataset['train']

    def get_dev_examples(self, data_dir):
        return self.dataset['validation']

    def get_test_examples(self, data_dir):
        return self.dataset['test']

    def get_output_mode(self):
        return self.dataset.metadata['output_mode']

    def get_num_tasks(self):
        return len(self.dataset.metadata['tasks'])

    def get_labels(self):
        return self.labels
    
    def get_classifiers(self):
        return self.classifiers

"""
Add processor classes for new tasks here.
"""
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
                   'tlinkx-nc': Thyme1ContainsRelationProcessor,
                   'tlinkx': Thyme1AllRelationProcessor,
                   'dphe': DpheProcessor,
                   'i2b22008': i2b22008Processor,
                   'ucidrug': UciDrugSentimentProcessor,
                   'mimic_radi': MimicRadiProcessor,
                   'mimic_3': Mimic_3_Processor,
                   'mimic_7': Mimic_7_Processor,
                   'covid': CovidProcessor,
                   'psych_domain': PsychDomainProcessor,
                   'infer': InferringProcessor
                  }



"""
Add output modes for new tasks here.
"""
# cnlp_output_modes = {'polarity': classification,
#                 'uncertainty': classification,
#                 'history': classification,
#                 'dtr': classification,
#                 'alink': classification,
#                 'alinkx': classification,
#                 'tlink': classification,
#                 'nc': classification,
#                 'timecat': classification,
#                 'conmod': classification,
#                 'timex': tagging,
#                 'event': tagging,
#                 'dphe': tagging,
#                 'tlinkx-nc': relex,
#                 'tlinkx': relex,
#                 'i2b22008': mtl,
#                 'ucidrug': classification,
#                 'mimic_radi': mtl,
#                 'mimic_3': classification,
#                 'mimic_7': classification,
#                 'covid': classification,
#                 'psych_domain': mtl
#                 }


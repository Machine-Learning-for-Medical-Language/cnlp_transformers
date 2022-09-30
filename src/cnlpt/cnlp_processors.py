"""
Module containing auto processor class, which infers labels, task type, and output
modes for tasks and datasets that use a few conventional formats.

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


class AutoProcessor(DataProcessor):
    """
    A special type of processor that tries to infer the details about the dataset from the
    artifacts that are present in the data directory.

    TODO - add documentation of the expected file formats for json and csv defaults
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

        if ext_check_file.endswith('csv') or ext_check_file.endswith('tsv'):
            self.dataset = load_dataset('csv', data_files=data_files, sep='\t', column_names=['label', 'text'])
            metadata = {'tasks': ['label']}
        elif ext_check_file.endswith('json'):
            self.dataset = load_dataset('json', data_files=data_files, field='data')
            with open(join(data_dir, ext_check_file), 'rt', encoding="utf-8") as f:
                json_file = json.load(f)
                metadata = json_file['metadata']
        else:
            raise ValueError('Data file %s has an extension that we cannot handle (tried csv and json)' % (train_file))

        self.dataset.metadata = metadata

        any_split = next(iter(self.dataset.values()))

        ## FIXME - this will assume that all tasks have the same labelset
        unique_labels = list(set( any_split[self.dataset.metadata['tasks'][0]]) )

        # Probably a reasonable default, and then we'll check for the other cases
        output_mode = classification

        ## Check if any unique label has a space in it, then we know we are actually 
        ## dealing with a tagging dataset (FIXME to expand logic to handle relations which ## also have spaces as well as other characters)
        for label in unique_labels:
            if label[-1] == ')':
                output_mode = relex
                break
            elif ' ' in str(label):
                assert 'output_mode' not in metadata or metadata['output_mode'] == tagging, 'Output mode is ambiguous because we inferred tagging due to spaces in labels, but data file has another output mode.'
                output_mode = tagging
                break
        
        
        ## get the complete set of unique tags by splitting each set of tags seen so far
        if output_mode == tagging:
            unique_tags = set()
            for label in unique_labels:
                tags = label.split(' ')
                unique_tags.update(tags)
            unique_labels = list(unique_tags)
        elif output_mode == relex:
            unique_relations = set()
            for label in unique_labels:
                inst_rels = label.split(' , ')
                for rel in inst_rels:
                    rel_cat = rel.split(',')[-1]
                    if rel_cat[-1] == ')':
                        rel_cat = rel_cat[:-1]
                    unique_relations.add(rel_cat)
            unique_labels = list(unique_relations)


        metadata['output_mode'] = output_mode
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

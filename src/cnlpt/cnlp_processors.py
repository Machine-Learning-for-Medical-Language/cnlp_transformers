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
from typing import Callable, Dict, Optional, List, Union, Any, Set
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

class AutoProcessor(DataProcessor):
    """
    A special type of processor that tries to infer the details about the dataset from the
    artifacts that are present in the data directory.

    TODO - add documentation of the expected file formats for json and csv defaults
    """
    def __init__(self, data_dir:str, tasks:Set[str]=None):
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
            if ext_check_file.endswith('csv'):
                sep = ','
            else:
                sep = '\t'

            self.dataset = load_dataset('csv', sep=sep, data_files=data_files)

            ## find out what tasks are available to this dataset, and see the overlap with what the
            ## user specified at the cli, remove those tasks so we don't also get them from other datasets
            ## and overwrite these.
            first_split = next(iter(self.dataset.values()))
            dataset_tasks = tasks.intersection(first_split.features.keys())
            tasks -= dataset_tasks
            dataset_tasks = list(dataset_tasks)
            dataset_tasks.sort()
            metadata = {'tasks': dataset_tasks}
        elif ext_check_file.endswith('json'):
            self.dataset = load_dataset('json', data_files=data_files, field='data')
            with open(join(data_dir, ext_check_file), 'rt', encoding="utf-8") as f:
                json_file = json.load(f)
                metadata = json_file['metadata']
                output_mode = metadata['output_mode']
            if not tasks is None:
                dataset_tasks = tasks.intersection(metadata['tasks'])
                tasks -= dataset_tasks
                dataset_tasks = list(dataset_tasks)
                dataset_tasks.sort()
                metadata['tasks'] = dataset_tasks
        else:
            raise ValueError('Data file %s has an extension that we cannot handle (tried csv and json)' % (train_file))

        self.dataset.metadata = metadata

        any_split = next(iter(self.dataset.values()))

        metadata['output_mode'] = []
        self.labels = []
        for task_ind,dataset_task in enumerate(metadata['tasks']):
            # Probably a reasonable default, and then we'll check for the other cases
            output_mode = classification

            unique_labels = list(set( any_split[self.dataset.metadata['tasks'][task_ind]]) )

            ## Check if any unique label has a space in it, then we know we are actually 
            ## dealing with a tagging dataset, or if it ends in ), in which case it is a relation task.
            for label in unique_labels:
                if str(label)[-1] == ')':
                    output_mode = relex
                    break
                elif ' ' in str(label):
                    assert 'output_mode' not in metadata or len(metadata['output_mode']) <= task_ind or metadata['output_mode'] == tagging, 'Output mode is ambiguous because we inferred tagging due to spaces in labels, but data file has another output mode.'
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


            metadata['output_mode'].append(output_mode)
            unique_labels.sort()
            self.labels.append(unique_labels)

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

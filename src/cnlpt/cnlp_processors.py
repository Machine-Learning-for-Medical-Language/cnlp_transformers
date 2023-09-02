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
import datasets
from datasets import load_dataset
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
import numpy  # for Sphinx
from tqdm import tqdm

logger = logging.getLogger(__name__)

mtl = "mtl"
classification = "classification"
tagging = "tagging"
relex = "relations"


def get_unique_labels(
    dataset, tasks: List[str], task_output_modes: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Return the set of unique labels defined in a dataset by iterating through the dataset.
    :param tasks: List of tasks that the caller cares about
    :param task_output_modes: Dictionary mapping from task names to task output mode
    :return: Dictionary from task names to a list of unique labels for that task
    """
    dataset_unique_labels = dict()
    for task_ind, task_name in enumerate(tasks):
        unique_labels = set()
        # check all splits for labels just in case they do not fully overlap
        for split in dataset:
            # Add labels from this split to the overall label set and give a warning if they are not the same
            split_labels = set(dataset[split][task_name])
            unique_labels |= split_labels

        unique_labels = list(unique_labels)

        output_mode = task_output_modes[task_name]

        ## get the complete set of unique tags by splitting each set of tags seen so far
        if output_mode == tagging:
            unique_tags = set()
            for label in unique_labels:
                tags = label.split(" ")
                unique_tags.update(tags)
            unique_labels = list(unique_tags)
        elif output_mode == relex:
            unique_relations = set()
            for label in unique_labels:
                inst_rels = label.split(" , ")
                for rel in inst_rels:
                    rel_cat = rel.split(",")[-1]
                    if rel_cat[-1] == ")":
                        rel_cat = rel_cat[:-1]
                    unique_relations.add(rel_cat)
            unique_labels = list(unique_relations)

        unique_labels.sort()

        dataset_unique_labels[task_name] = unique_labels

    return dataset_unique_labels


def infer_output_modes(dataset: datasets.DatasetDict) -> Dict[str, str]:
    """
    Figure out what output mode each task in the dataset requires by looking at the format of the labels.
    :param dataset: HF datasets DatasetDict containing the loaded dataset
    :return: Dictionary mapping from task names to output modes
    """
    task_output_modes = {}
    for task_ind, task_name in enumerate(dataset.tasks):
        output_mode = classification
        unique_labels = set()
        # check all splits for labels just in case they do not fully overlap
        for split in dataset:
            # Add labels from this split to the overall label set and give a warning if they are not the same
            split_labels = set(dataset[split][task_name])
            unique_labels |= split_labels

        unique_labels = list(unique_labels)

        ## Check if any unique label has a space in it, then we know we are actually
        ## dealing with a tagging dataset, or if it ends in ), in which case it is a relation task.
        for label in unique_labels:
            if str(label)[-1] == ")":
                output_mode = relex
                break
            elif " " in str(label):
                output_mode = tagging
                break

        task_output_modes[task_name] = output_mode

    return task_output_modes


def get_task_pruned_dataset(
    dataset: datasets.DatasetDict, tasks: List[str], unique_labels: Dict[str, List[str]]
) -> datasets.DatasetDict:
    """
    Remove tasks from the dataset that only have 1 unique label
    """
    tasks_to_remove = []
    for task_ind, task_name in enumerate(tasks):
        if len(unique_labels[task_name]) == 1:
            logger.warn(
                "Task named %s has only 1 unique label -- this column from the data"
                % (task_name)
            )
            tasks_to_remove.append(task_name)

    for split in dataset.keys():
        dataset[split] = dataset[split].remove_columns(tasks_to_remove)

    for task in tasks_to_remove:
        tasks.remove(task)
        unique_labels.pop(task)

    return dataset


class AutoProcessor(DataProcessor):
    """
    A special type of processor that tries to infer the details about the dataset from the
    artifacts that are present in the data directory.

    TODO - add documentation of the expected file formats for json and csv defaults
    """

    def __init__(self, data_dir: str, tasks: Set[str] = None, max_train_items=-1):
        super().__init__()

        train_file = dev_file = test_file = None
        data_files = {}
        for fn in os.listdir(data_dir):
            if fn.startswith("train"):
                train_file = fn
                data_files["train"] = join(data_dir, train_file)
            elif fn.startswith("dev") or fn.startswith("valid"):
                dev_file = fn
                data_files["validation"] = join(data_dir, dev_file)
            elif fn.startswith("test"):
                test_file = fn
                data_files["test"] = join(data_dir, test_file)

        if train_file is None and dev_file is None and test_file is None:
            raise ValueError("This dataset doesn't have train, dev, or test files")

        metadata = None
        if train_file is not None:
            ext_check_file = train_file
        elif dev_file is not None:
            ext_check_file = dev_file
        else:
            ext_check_file = test_file

        if ext_check_file.endswith("csv") or ext_check_file.endswith("tsv"):
            if ext_check_file.endswith("csv"):
                sep = ","
            else:
                sep = "\t"

            self.dataset = load_dataset("csv", sep=sep, data_files=data_files)

            ## find out what tasks are available to this dataset, and see the overlap with what the
            ## user specified at the cli, remove those tasks so we don't also get them from other datasets
            ## and overwrite these.
            first_split = next(iter(self.dataset.values()))
            dataset_tasks = first_split.features.keys() - set(
                ["text", "text_a", "text_b"]
            )
            if tasks is None:
                active_tasks = list(dataset_tasks)
            else:
                active_tasks = set(tasks).intersection(dataset_tasks)
                active_tasks = list(active_tasks)
            active_tasks.sort()
            self.dataset.task_output_modes = {}
        elif ext_check_file.endswith("json"):
            self.dataset = load_dataset("json", data_files=data_files, field="data")
            with open(join(data_dir, ext_check_file), "rt", encoding="utf-8") as f:
                json_file = json.load(f)
                if "metadata" in json_file:
                    metadata = json_file["metadata"]
                elif os.path.exists(join(data_dir, "metadata.json")):
                    with open(join(data_dir, "metadata.json")) as mf:
                        metadata = json.load(mf)
                else:
                    raise Exception(
                        "No metadata was available in the data file or in the same directory!"
                    )

                dataset_task2output = {}
                for subtask in metadata["subtasks"]:
                    dataset_task2output[subtask["task_name"]] = subtask["output_mode"]

            dataset_tasks = list(dataset_task2output.keys())
            if tasks is None:
                active_tasks = set(dataset_task2output.keys())
            else:
                active_tasks = set(tasks).intersection(set(dataset_task2output.keys()))

            active_tasks = list(active_tasks)
            active_tasks.sort()

            self.dataset.task_output_modes = dataset_task2output
        else:
            raise ValueError(
                "Data file %s has an extension that we cannot handle (tried csv and json)"
                % (train_file)
            )

        logger.info("This dataset contains these tasks: %s" % (str(dataset_tasks)))
        logger.info("These tasks overlap with user input: %s" % (str(active_tasks)))

        self.dataset.tasks = active_tasks
        if len(self.dataset.task_output_modes) == 0:
            self.dataset.task_output_modes = infer_output_modes(self.dataset)

        # convert label columns to strings
        logger.info("Converting columns to strings")
        for task in tqdm(self.dataset.tasks):
            if self.dataset.task_output_modes[task] == classification:
                task_str = task + "_str"
                for split in self.dataset:
                    # create a new column casting every element to string, remove old (int) column, rename new (str) column
                    self.dataset[split] = self.dataset[split].add_column(
                        task_str, [str(x) for x in self.dataset[split][task]]
                    )
                    self.dataset[split] = self.dataset[split].remove_columns(task)
                    self.dataset[split] = self.dataset[split].rename_column(
                        task_str, task
                    )

        # get any split of the data and ask for the set of unique labels for each task in the dataset from that split
        self.labels = get_unique_labels(
            self.dataset, self.dataset.tasks, self.dataset.task_output_modes
        )

        self.dataset = get_task_pruned_dataset(
            self.dataset, self.dataset.tasks, self.labels
        )

        self.classifiers = self.dataset.tasks

        if max_train_items > 0:
            self.dataset["train"] = self.dataset["train"].select(range(max_train_items))

        print("Loaded dataset has length %d" % (len(self.dataset)))

    def get_train_examples(self):
        return self.dataset["train"]

    def get_dev_examples(self):
        return self.dataset["validation"]

    def get_test_examples(self):
        return self.dataset["test"]

    def get_output_mode(self, task_name):
        return self.dataset.task_output_modes[task_name]

    def get_output_modes(self):
        return self.dataset.task_output_modes

    def get_num_tasks(self):
        return len(self.dataset.tasks)

    def get_labels(self):
        return self.labels

    def get_classifiers(self):
        return self.classifiers

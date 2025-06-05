import logging
from typing import Union

import datasets
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from ....args import CnlpDataArguments
from ...data.cnlp_datasets import AutoProcessor
from ...data.tasks import CLASSIFICATION, RELEX, TAGGING, TaskType
from ..features import HierarchicalInputFeatures, InputFeatures
from ..preprocess import cnlp_preprocess_data, none_column

text_columns = ["text", "text_a", "text_b"]

logger = logging.getLogger(__name__)


class ClinicalNlpDataset(Dataset):
    """
    Copy-pasted from GlueDataset with glue task-specific code changed;
    moved into here to be self-contained.

    :param args: the data training args for this experiment
    :param tokenizer: the tokenizer
    :param limit_length: if provided, the number of
        examples to include in the dataset
    :param cache_dir: if provided, the directory to save/load a cache
        of this dataset
    :param hierarchical: whether to structure the data for the hierarchical
        model (:class:`cnlpt.HierarchicalTransformer.HierarchicalModel`)
    """

    args: CnlpDataArguments
    features: list[InputFeatures]

    def __init__(
        self,
        args: CnlpDataArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Union[int, None] = None,
        cache_dir: Union[str, None] = None,
        hierarchical: bool = False,
    ):
        self.args = args
        self.class_weights = None
        self.tasks_to_labels = {}
        self.hierarchical = hierarchical
        self.datasets = []
        self.processed_dataset = []
        self.output_modes: dict[str, TaskType] = {}

        # Load data features from cache or dataset file
        self.label_lists = []
        self.num_train_instances = 0

        if self.hierarchical:
            assert self.args.chunk_len is not None and self.args.num_chunks is not None
            implicit_max_len = self.args.chunk_len * self.args.num_chunks
            if self.args.max_seq_length < implicit_max_len:
                raise ValueError(
                    "For the hierarchical model, the max seq length should be equal to the chunk length * num_chunks, otherwise what is the point?"
                )

        # if cli supplies no tasks, the processor will assume we want all the tasks, but we do need to have a conventional order
        # for the model to use, so we need to still create a tasks variable.
        tasks = None if args.task_name is None else list(args.task_name)
        for data_dir in args.data_dir:
            dataset_processor = AutoProcessor(
                data_dir, tasks, max_train_items=args.max_train_items
            )

            # Make sure that any overlapping task names have the same label sets and output mode definitions
            self._reconcile_labels_lists(dataset_processor)
            self._reconcile_output_modes(dataset_processor)

            self.datasets.append(dataset_processor.dataset)

        ## Each processed dataset will have been pruned to remove columns that the user _didn't_ ask for,
        ## but we need to add columns for tasks that it doesn't have that the user did ask for, with the
        ## appropriate task types and  special values so pytorch knows not to compute losses over those
        ## tasks for those inputs. Need to  do that after we've read all datasets and inferred all the task
        ## types and label spaces.
        self._reconcile_columns()
        combined_dataset = self._concatenate_datasets()

        if tasks is None:
            # i.e., cli did not supply any task ordering:
            if "train" in combined_dataset:
                split = "train"
            elif "dev" in combined_dataset:
                split = "dev"
            else:
                split = "test"

            tasks = list(
                combined_dataset[split].features.keys() - {"text", "text_a", "text_b"}
            )

        self.tasks = tasks

        if self.args.character_level:
            logger.warning(
                "No real implementation for character level event masking yet, using a placeholder"
            )
        self.processed_dataset = combined_dataset.map(
            cnlp_preprocess_data,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset, organizing labels, creating hierarchical segments if necessary",
            batch_size=100,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": args.max_seq_length,
                "label_lists": self.tasks_to_labels,
                "output_modes": self.output_modes,
                "inference": "train" not in combined_dataset,
                "hierarchical": self.hierarchical,
                "chunk_len": self.args.chunk_len,
                "num_chunks": self.args.num_chunks,
                "insert_empty_chunk_at_beginning": self.args.insert_empty_chunk_at_beginning,
                "truncate_examples": self.args.truncate_examples,
                "character_level": self.args.character_level,
                "tasks": tasks,
            },
        )

        if args.max_eval_items > 0:
            new_validation = self.processed_dataset["validation"].train_test_split(
                test_size=args.max_eval_items
            )["test"]
            self.processed_dataset["validation"] = new_validation

        self.num_train_instances += self.processed_dataset["train"].num_rows

    def _reconcile_labels_lists(self, processor: AutoProcessor):
        """
        given a new data processor, which extracted a label list for every task it contained,
        we reconcile it with existing label list for the same task
        :param processor: An AutoProcessor object that contains a processed dataset
        """
        for task, labels in processor.get_labels().items():
            if task in self.tasks_to_labels:
                ## a subset of existing labels (WARN), superset of existing labels (WARN), mixture, or no overlap (ERROR)
                new_labels = set(labels)
                old_labels = set(self.tasks_to_labels[task])
                if new_labels.isdisjoint(old_labels):
                    raise Exception(
                        f"The same task name has disjoint sets of labels in different dataset: {old_labels!s} vs. {new_labels!s}"
                    )
                elif new_labels != old_labels:
                    logger.warning(
                        f"Two different datasets have the same task name but not completely equal label lists: {old_labels!s} vs. {new_labels!s}. We will merge them."
                    )
                    self.tasks_to_labels[task] = list(old_labels.union(new_labels))
                else:
                    ## they match completely, nothing to be done
                    pass
            else:
                self.tasks_to_labels[task] = labels

    def _reconcile_output_modes(self, processor: AutoProcessor):
        """
        given a new data processor, which inferred output modes for its tasks, make
        sure those output modes agree with existing inferred output modes for any
        same-named tasks.
        """
        for task, output_mode in processor.get_output_modes().items():
            if task in self.output_modes:
                # There is an existing task with this name
                existing_output_mode = self.output_modes[task]
                if output_mode != existing_output_mode:
                    raise Exception(
                        f"The task {task} has two different output modes in different datasets and might not be the same task: {existing_output_mode} vs. {output_mode}"
                    )
            else:
                self.output_modes[task] = output_mode

    def _reconcile_columns(self):
        """
        The overall dataset should get the superset of all task columns, even if some of the tasks aren't in
        all of the component datasets
        """
        tasks = self.tasks_to_labels.keys()

        for dataset in self.datasets:
            for split_name in dataset.keys():
                for task in tasks:
                    if task not in dataset[split_name].column_names:
                        if self.output_modes[task] == TAGGING:
                            dataset[split_name] = dataset[split_name].add_column(
                                task, [none_column] * len(dataset[split_name])
                            )
                        elif self.output_modes[task] == RELEX:
                            pass
                        elif self.output_modes[task] == CLASSIFICATION:
                            dataset[split_name] = dataset[split_name].add_column(
                                task, [none_column] * len(dataset[split_name])
                            )

                for column in dataset[split_name].column_names:
                    if column not in tasks and column not in text_columns:
                        dataset[split_name] = dataset[split_name].remove_columns(column)

    def _concatenate_datasets(self) -> datasets.DatasetDict:
        """
        We have multiple dataset dicts, we need to create a single dataset dict
        where we concatenate each of the splits first.
        """
        datasets_by_split = {}
        for dataset in self.datasets:
            for split in dataset:
                if split not in datasets_by_split:
                    datasets_by_split[split] = []
                datasets_by_split[split].append(dataset[split])

        for split_name, split_data in datasets_by_split.items():
            datasets_by_split[split_name] = datasets.concatenate_datasets(split_data)

        return datasets.DatasetDict(datasets_by_split)

    def __len__(self) -> int:
        """
        Length method for this class.

        :return: the number of datasets included in this dataset
        """
        return len(self.datasets)

    def __getitem__(self, i) -> Union[InputFeatures, HierarchicalInputFeatures]:
        """
        Getitem method for this class.

        :param i: the index of the example to retrieve
        :return: the example at index `i`
        """
        return self.features[i]

    def get_labels(self) -> dict[str, list[str]]:
        """
        Retrieve the label lists for all the tasks for the dataset.

        :return: the dictionary of label lists indexed by task name
        """
        return self.tasks_to_labels

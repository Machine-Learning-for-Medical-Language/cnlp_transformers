import os
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
from datasets import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from .data_reader import CnlpDataReader
from .preprocess import preprocess_raw_data


@dataclass(frozen=True)
class HierarchicalDataConfig:
    chunk_len: int
    num_chunks: int
    prepend_empty_chunk: bool


def load_tokenizer(
    model_name_or_path: str,
    hf_cache_dir: str | None = None,
    truncation_side: Literal["left", "right"] = "right",
    character_level: bool = False,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=hf_cache_dir,
        add_prefix_space=True,
        truncation_side=truncation_side,
        additional_special_tokens=(
            ["<e>", "</e>", "<a1>", "</a1>", "<a2>", "</a2>", "<cr>", "<neg>"]
            if not character_level
            else None
        ),
    )
    return tokenizer


class TruncationSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class CnlpDataset:
    """A preprocessed dataset for clinical NLP."""

    def __init__(
        self,
        data_dir: str | os.PathLike,
        tokenizer: str | PreTrainedTokenizer = "roberta-base",
        task_names: list[str] | None = None,
        hier_config: HierarchicalDataConfig | None = None,
        truncation_side: TruncationSide = TruncationSide.RIGHT,
        max_seq_length: int = 128,
        use_data_cache: bool = True,
        max_train: int | None = None,
        max_eval: int | None = None,
        max_test: int | None = None,
        allow_disjoint_labels: bool = False,
        character_level: bool = False,
        hf_cache_dir: str | None = None,
    ):
        """Create a new `CnlpDataset`.

        Args:
            args: Arguments for data loading and preprocessing.
            tokenizer: Tokenizer to tokenize the raw data.
            hierarchical: Whether this data is being preprocessed for a hierarchical model. Defaults to False.
        """

        if hier_config is not None:
            implicit_max_len = hier_config.chunk_len * hier_config.num_chunks

            # TODO(ian) should this be `!=`` instead of `<`?
            if max_seq_length < implicit_max_len:
                raise ValueError(
                    "For the hierarchical model, the max seq length should be equal to the chunk length * num_chunks, otherwise what is the point?"
                )

        self.data_dir = data_dir
        if isinstance(tokenizer, str):
            self.tokenizer = load_tokenizer(
                tokenizer,
                hf_cache_dir=hf_cache_dir,
                truncation_side=truncation_side,
                character_level=character_level,
            )
        else:
            self.tokenizer = tokenizer

        reader = CnlpDataReader(allow_disjoint_labels=allow_disjoint_labels)
        reader.load_dir(data_dir)

        self.tasks = reader.get_tasks(task_names)
        self.dataset = reader.dataset

        if max_train is not None:
            self.dataset["train"] = self.dataset["train"].take(max_train)
        if max_eval is not None:
            self.dataset["validation"] = self.dataset["validation"].take(max_eval)
        if max_test is not None:
            self.dataset["test"] = self.dataset["test"].take(max_test)

        self.hier_config = hier_config
        self.truncation_side = truncation_side
        self.max_seq_length = max_seq_length
        self.character_level = character_level

        split_data: Dataset
        for split_name, split_data in self.dataset.items():
            self.dataset[split_name] = split_data.map(
                preprocess_raw_data,
                desc=f"Preprocessing {split_name} data",
                batched=True,
                load_from_cache_file=use_data_cache,
                batch_size=100,
                num_proc=1,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                    "tasks": self.tasks,
                    "max_length": self.max_seq_length,
                    "inference_only": "train" not in reader.split_names,
                    "character_level": self.character_level,
                    "hier_config": self.hier_config,
                },
            )

    @property
    def train_data(self):
        """This dataset's train split."""
        return self.dataset["train"]

    @property
    def validation_data(self):
        """This dataset's validation split."""
        return self.dataset["validation"]

    @property
    def test_data(self):
        """This dataset's test split."""
        return self.dataset["test"]

    def get_class_weights(self, device: torch.device):
        class_weights: dict[str, torch.FloatTensor] = {}
        for task in self.tasks:
            train_labels = self.train_data[task.name]
            weights: list[float] = []
            train_label_counts = Counter(train_labels)
            for label in task.labels:
                # class weights are determined by severity of class imbalance
                weights.append(
                    len(train_labels) / (len(task.labels) * train_label_counts[label])
                )

            class_weights[task.name] = torch.tensor(
                # if we just have the one class, simplify the tensor or pytorch will be mad
                # TODO(ian) why would we ever have just one class??
                weights[0] if len(weights) == 1 else weights
            ).to(device)

        return class_weights

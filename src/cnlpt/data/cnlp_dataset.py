from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from ..args.data_args import CnlpDataArguments
from .data_reader import CnlpDataReader
from .preprocess import preprocess_raw_data


def _validate_dataset_args(args: CnlpDataArguments, hierarchical: bool):
    if hierarchical:
        if args.chunk_len is None or args.num_chunks is None:
            raise ValueError(
                "For the hierarchical model, data_args.chunk_len and data_args.num_chunks must be specified."
            )
        implicit_max_len = args.chunk_len * args.num_chunks
        if args.max_seq_length < implicit_max_len:
            raise ValueError(
                "For the hierarchical model, the max seq length should be equal to the chunk length * num_chunks, otherwise what is the point?"
            )


class CnlpDataset:
    """A preprocessed dataset for clinical NLP."""

    def __init__(
        self,
        args: CnlpDataArguments,
        tokenizer: PreTrainedTokenizer,
        hierarchical: bool = False,
    ):
        """Create a new `CnlpDataset`.

        Args:
            args: Arguments for data loading and preprocessing.
            tokenizer: Tokenizer to tokenize the raw data.
            hierarchical: Whether this data is being preprocessed for a hierarchical model. Defaults to False.
        """
        _validate_dataset_args(args, hierarchical)

        self.hierarchical = hierarchical

        reader = CnlpDataReader(allow_disjoint_labels=args.allow_disjoint_labels)
        for data_dir in args.data_dir:
            reader.load_dir(data_dir)

        self.tasks = reader.get_tasks(args.task_name or None)
        self.dataset = reader.dataset

        if (val_limit := (args.max_eval_items or 0)) > 0:
            self.dataset["validation"] = self.dataset["validation"].take(val_limit)

        if (train_limit := (args.max_train_items or 0)) > 0:
            self.dataset["train"] = self.dataset["train"].take(train_limit)

        if (test_limit := (args.max_test_items or 0)) > 0:
            self.dataset["test"] = self.dataset["test"].take(test_limit)

        split_data: Dataset
        for split_name, split_data in self.dataset.items():
            self.dataset[split_name] = split_data.map(
                preprocess_raw_data,
                desc=f"Preprocessing {split_name} data",
                batched=True,
                load_from_cache_file=not args.overwrite_cache,
                batch_size=100,
                num_proc=1,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "tasks": self.tasks,
                    "max_length": args.max_seq_length,
                    "inference_only": "train" not in reader.split_names,
                    "hierarchical": self.hierarchical,
                    "character_level": args.character_level,
                    "chunk_len": args.chunk_len,
                    "num_chunks": args.num_chunks,
                    "insert_empty_chunk_at_beginning": args.insert_empty_chunk_at_beginning,
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

import functools
import operator

from datasets import DatasetDict, IterableDatasetDict
from torch.utils.data.dataset import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils import PreTrainedTokenizer

from ...args import DaptArguments
from ..cnlp_datasets import AutoProcessor


def group_texts(chunk_size, examples):
    # Concatenate all texts
    concatenated_examples = {
        k: functools.reduce(operator.iadd, examples[k], []) for k in examples.keys()
    }
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[next(iter(examples.keys()))])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_fn(tokenizer, examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


class DaptDataset(Dataset):
    def __getitem__(self, index):
        return self.train[index]

    def __init__(
        self,
        args: DaptArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        self.args = args
        self.tokenizer = tokenizer

        processor = AutoProcessor(self.args.data_dir, tasks=None)

        # This can probably be refined
        dataset: DatasetDict = processor.dataset
        remove_columns = {"text", "id", *processor.get_labels()}.intersection(
            set(dataset.column_names["train"])
        )

        dataset = dataset.map(
            functools.partial(tokenize_fn, self.tokenizer),
            batched=True,
            remove_columns=list(remove_columns),
        )
        dataset = dataset.map(
            functools.partial(group_texts, self.args.chunk_size),
            batched=True,
        )

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)) or args.no_eval:
            self.dataset = dataset
        else:
            self.dataset = dataset.train_test_split(
                test_size=args.test_size,
                seed=args.seed,
            )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=self.args.mlm_probability
        )

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def test(self):
        return self.dataset["test"]

from dataclasses import dataclass, field
from typing import Union


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using :class:`transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: list[str] = field(
        metadata={
            "help": "The input data dirs. A space-separated list of directories that "
            "should contain the .tsv files (or other data files) for the task. "
            "Should be presented in the same order as the task names."
        }
    )

    task_name: list[str] = field(
        default_factory=lambda: None,
        metadata={
            "help": "A space-separated list of tasks to train on (mainly used as keys to internally track and display output)"
        },
    )
    # field(
    #     metadata={"help": "A space-separated list of tasks to train on: " + ", ".join(cnlp_processors.keys())})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    weight_classes: bool = field(
        default=False,
        metadata={
            "help": "A flag that indicates whether class-specific loss should be used. "
            "This can be useful in cases with severe class imbalance. The formula "
            "for a weight of a class is the count of that class divided the count "
            "of the rarest class."
        },
    )

    chunk_len: Union[int, None] = field(
        default=None, metadata={"help": "Chunk length for hierarchical model"}
    )
    character_level: bool = field(
        default=False,
        metadata={
            "help": "Whether the dataset sould be processed at the character level"
            "(otherwise will be processed at the token level)"
        },
    )

    num_chunks: Union[int, None] = field(
        default=None, metadata={"help": "Max chunk count for hierarchical model"}
    )

    insert_empty_chunk_at_beginning: bool = field(
        default=False,
        metadata={"help": "Whether to insert an empty chunk for hierarchical model"},
    )

    truncate_examples: bool = field(
        default=False,
        metadata={
            "help": "Whether to truncate input examples when displaying them in the log"
        },
    )

    max_train_items: Union[int, None] = field(
        default=-1,
        metadata={
            "help": "Set a number of train instances to use during training (useful for debugging data processing logic if a dataset is very large. Default is to train on all training data."
        },
    )

    max_eval_items: Union[int, None] = field(
        default=-1,
        metadata={
            "help": "Set a number of validation instances to use during training (useful if a dataset has been created using dumb logic like 80/10/10 and 10%% takes forever to evaluate on. Default is evaluate on all validation data."
        },
    )

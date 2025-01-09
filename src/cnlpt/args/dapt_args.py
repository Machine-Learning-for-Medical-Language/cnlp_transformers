from dataclasses import dataclass, field
from typing import Union


@dataclass
class DaptArguments:
    encoder_name: Union[str, None] = field(
        default="roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    output_dir: Union[str, None] = field(
        default=None, metadata={"help": "Directory path to write trained model to."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    data_dir: Union[str, None] = field(
        default=None, metadata={"help": "The data dir for domain-adaptive pretraining."}
    )
    cache_dir: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    chunk_size: int = field(
        default=128,
        metadata={"help": "The chunk size for domain-adaptive pretraining."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={
            "help": "The token masking probability for domain-adaptive pretraining."
        },
    )
    test_size: float = field(
        default=0.2,
        metadata={"help": "The test split proportion for domain-adaptive pretraining."},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "The random seed to use for a train/test split for domain-adaptive pretraining (requires --dapt-encoder)."
        },
    )
    no_eval: bool = field(
        default=False,
        metadata={"help": "Don't split into train and test; just pretrain."},
    )

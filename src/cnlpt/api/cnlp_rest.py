import logging
import os

import torch
from datasets import Dataset
from pydantic import BaseModel
from typing import List, Literal

# Modeling imports
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import torch.backends
import torch.backends.mps
import logging
from transformers.tokenization_utils import PreTrainedTokenizer

from ..cnlp_data import cnlp_preprocess_data
from ..CnlpModelForClassification import CnlpConfig


class UnannotatedDocument(BaseModel):
    doc_text: str


class EntityDocument(BaseModel):
    """doc_text: The raw text of the document
    offset:  A list of entities, where each is a tuple of character offsets into doc_text for that entity
    """

    doc_text: str
    entities: list[list[int]]


def create_dataset(
    inst_list: list[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    hier: bool = False,
    chunk_len: int = 200,
    num_chunks: int = 40,
    insert_empty_chunk_at_beginning: bool = False,
):
    """Use a tokenizer to create a dataset from a list of strings."""
    dataset = Dataset.from_dict({"text": inst_list})
    task_dataset = dataset.map(
        cnlp_preprocess_data,
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset, organizing labels, creating hierarchical segments if necessary",
        batch_size=100,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "inference": True,
            "hierarchical": hier,
            # TODO: need to get this from the model if necessary
            "chunk_len": chunk_len,
            "num_chunks": num_chunks,
            "insert_empty_chunk_at_beginning": insert_empty_chunk_at_beginning,
            "truncate_examples": True,
        },
    )
    return task_dataset


def create_instance_string(doc_text: str, offsets: list[int]):
    start = max(0, offsets[0] - 100)
    end = min(len(doc_text), offsets[1] + 100)
    raw_str = (
        doc_text[start : offsets[0]]
        + " <e> "
        + doc_text[offsets[0] : offsets[1]]
        + " </e> "
        + doc_text[offsets[1] : end]
    )
    return raw_str.replace("\n", " ")


def resolve_device(
    device: Literal["cuda", "mps", "cpu", "auto"],
) -> Literal["cuda", "mps", "cpu"]:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "Device is set to 'cuda' but was not available; setting to 'cpu' and proceeding. If you have a GPU you need to debug why pytorch cannot see it."
        )
        device = "cpu"
    elif device == "mps" and not torch.mps.is_available():
        logging.warning(
            "Device is set to 'mps' but was not available; setting to 'cpu' and proceeding. If you have a GPU you need to debug why pytorch cannot see it."
        )
        device = "cpu"
    return device


def initialize_cnlpt_model(
    model_name,
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto",
    batch_size=8,
):
    args = [
        "--output_dir",
        "save_run/",
        "--per_device_eval_batch_size",
        str(batch_size),
        "--do_predict",
        "--report_to",
        "none",
    ]
    parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses(args=args)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    model = AutoModel.from_pretrained(
        model_name, cache_dir=os.getenv("HF_CACHE"), config=config
    )

    model = model.to(resolve_device(device))

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=None,
    )

    return tokenizer, trainer


def initialize_hier_model(
    model_name,
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto",
):
    config: CnlpConfig = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)

    model = AutoModel.from_pretrained(
        model_name, cache_dir=os.getenv("HF_CACHE"), config=config
    )
    model.train(False)

    model = model.to(resolve_device(device))

    return tokenizer, model

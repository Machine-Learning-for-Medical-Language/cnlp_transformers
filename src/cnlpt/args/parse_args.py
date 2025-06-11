import os
import sys
from typing import Any, Union, cast

import torch
from transformers.hf_argparser import DataClassType, HfArgumentParser

from .data_args import CnlpDataArguments
from .log import logger
from .model_args import CnlpModelArguments
from .training_args import CnlpTrainingArguments


def _cast_dataclasses_to_args(
    dataclasses: tuple[Any, ...],
) -> tuple[CnlpModelArguments, CnlpDataArguments, CnlpTrainingArguments]:
    return cast(
        tuple[CnlpModelArguments, CnlpDataArguments, CnlpTrainingArguments], dataclasses
    )


def _get_args_parser():
    args_dataclasses = cast(
        tuple[DataClassType, ...],
        (CnlpModelArguments, CnlpDataArguments, CnlpTrainingArguments),
    )
    return HfArgumentParser(args_dataclasses, prog="cnlpt train")


def parse_args_dict(
    args: dict[str, Any],
):
    return _cast_dataclasses_to_args(_get_args_parser().parse_dict(args))


def parse_args_json_file(
    json_file: Union[str, os.PathLike],
):
    return _cast_dataclasses_to_args(_get_args_parser().parse_json_file(json_file))


def parse_args_from_argv(
    argv: Union[list[str], None] = None,
):
    if argv is None:
        argv = sys.argv
    if len(argv) == 2 and argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        return parse_args_json_file(argv[1])
    else:
        return _cast_dataclasses_to_args(
            _get_args_parser().parse_args_into_dataclasses(argv)
        )


def preprocess_args(
    model_args: CnlpModelArguments,
    data_args: CnlpDataArguments,
    training_args: CnlpTrainingArguments,
):
    if (
        training_args.output_dir is not None
        and os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    labels = training_args.model_selection_label
    if isinstance(labels, list) and any(isinstance(item, int) for item in labels):
        logger.warning(
            f"It is not recommended to use ints as model selection labels: {tuple([item for item in labels if isinstance(item, int)])}. Labels should be input in string form."
        )

    if training_args.truncation_side_left:
        if model_args.model == "hier":
            logger.warning(
                "truncation_side_left flag is not available for the hierarchical model -- setting to false"
            )
            training_args.truncation_side_left = False

    if torch.mps.is_available():
        # pin_memory is unsupported on MPS, but defaults to True,
        # so we'll explicitly turn it off to avoid a warning.
        training_args.dataloader_pin_memory = False

    return model_args, data_args, training_args

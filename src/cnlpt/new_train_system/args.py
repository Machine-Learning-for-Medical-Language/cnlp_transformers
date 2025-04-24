import os
import sys
from typing import Any, Union

from transformers.hf_argparser import HfArgumentParser

from ..args import CnlpTrainingArguments, DataTrainingArguments, ModelArguments
from .log import logger


def _get_args_parser():
    return HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CnlpTrainingArguments)
    )


def parse_args_dict(
    args: dict[str, Any],
) -> tuple[ModelArguments, DataTrainingArguments, CnlpTrainingArguments]:
    return _get_args_parser().parse_dict(args)


def parse_args_json_file(
    json_file: Union[str, os.PathLike],
) -> tuple[ModelArguments, DataTrainingArguments, CnlpTrainingArguments]:
    return _get_args_parser().parse_json_file(json_file)


def parse_args_from_argv(
    argv: Union[list[str], None] = None,
) -> tuple[ModelArguments, DataTrainingArguments, CnlpTrainingArguments]:
    if argv is None:
        argv = sys.argv
    if len(argv) == 2 and argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        return parse_args_json_file(argv[1])
    else:
        return _get_args_parser().parse_args_into_dataclasses(argv)


def validate_args(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: CnlpTrainingArguments,
):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if training_args.model_selection_label is not None and any(
        isinstance(item, int) for item in training_args.model_selection_label
    ):
        logger.warning(
            f"It is not recommended to use ints as model selection labels: {tuple([item for item in training_args.model_selection_label if isinstance(item, int)])}. Labels should be input in string form."
        )

    if training_args.truncation_side_left:
        if model_args.model == "hier":
            logger.warning(
                "truncation_side_left flag is not available for the hierarchical model -- setting to right"
            )
            training_args.truncation_size_left = False

    return model_args, data_args, training_args

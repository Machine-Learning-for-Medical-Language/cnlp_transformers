"""
Domain-adaptive pretraining (see DAPT.md for details)
"""

import logging
import os
import sys
from typing import Any, Union

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .dapt_args import DaptArguments
from .data.cnlp_datasets import DaptDataset

logger = logging.getLogger(__name__)


def main(
    json_file: Union[str, None] = None, json_obj: Union[dict[str, Any], None] = None
):
    """
    Domain-adaptive pretraining.

    See :class:`cnlpt.cnlp_data.DaptArguments` for command-line arguments.

    :param typing.Optional[str] json_file: if passed, a path to a JSON file
        to use as the model, data, and training arguments instead of
        retrieving them from the CLI (mutually exclusive with ``json_obj``)
    :param typing.Optional[dict] json_obj: if passed, a JSON dictionary
        to use as the model, data, and training arguments instead of
        retrieving them from the CLI (mutually exclusive with ``json_file``)
    :rtype: typing.Dict[str, typing.Dict[str, typing.Any]]
    :return: the evaluation results (will be empty if ``--do_eval`` not passed)
    """
    parser = HfArgumentParser((DaptArguments,))
    dapt_args: DaptArguments

    if json_file is not None and json_obj is not None:
        raise ValueError("cannot specify json_file and json_obj")

    if json_file is not None:
        (dapt_args,) = parser.parse_json_file(json_file=json_file)
    elif json_obj is not None:
        (dapt_args,) = parser.parse_dict(json_obj)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (dapt_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (dapt_args,) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(dapt_args.output_dir)
        and os.listdir(dapt_args.output_dir)
        and not dapt_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({dapt_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
    #     (training_args.local_rank,
    #     training_args.device,
    #     training_args.n_gpu,
    #     bool(training_args.local_rank != -1),
    #     training_args.fp16)
    # )
    # logger.info("Training/evaluation parameters %s" % training_args)
    # logger.info("Data parameters %s" % data_args)
    # logger.info("Model parameters %s" % model_args)

    logger.info(f"Domain adaptation parameters {dapt_args}")

    # Set seed
    set_seed(dapt_args.seed)

    # Load tokenizer: Need this first for loading the datasets
    tokenizer = AutoTokenizer.from_pretrained(
        (
            dapt_args.tokenizer_name
            if dapt_args.tokenizer_name
            else dapt_args.encoder_name
        ),
        cache_dir=dapt_args.cache_dir,
        add_prefix_space=True,
        # additional_special_tokens=['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
    )

    model = AutoModelForMaskedLM.from_pretrained(dapt_args.encoder_name)

    dataset = DaptDataset(dapt_args, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=dapt_args.output_dir),
        train_dataset=dataset.train,
        eval_dataset=dataset.test if not dapt_args.no_eval else None,
        data_collator=dataset.data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # write model out?
    trainer.save_model()


if __name__ == "__main__":
    main()

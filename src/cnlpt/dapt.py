"""
Domain-adaptive pretraining (see DAPT.md for details)
"""

import logging
import os
import sys
from typing import Any, Union

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_utils import PreTrainedModel

from .CnlpModelForClassification import CnlpConfig, generalize_encoder_forward_kwargs
from .cnlp_args import DaptArguments
from .cnlp_data import DaptDataset

logger = logging.getLogger(__name__)


class DaptModel(PreTrainedModel):
    base_model_prefix = "cnlpt"
    config_class = CnlpConfig

    def __init__(
        self,
        config: config_class,
    ):
        super().__init__(config)
        encoder_config = AutoConfig.from_pretrained(config._name_or_path)
        encoder_config.vocab_size = config.vocab_size
        config.encoder_config = encoder_config.to_dict()
        model = AutoModelForMaskedLM.from_config(encoder_config)
        self.encoder = model.from_pretrained(config._name_or_path)
        # if not config.character_level:
        self.encoder.resize_token_embeddings(encoder_config.vocab_size)

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            labels,
    ):
        kwargs = generalize_encoder_forward_kwargs(
            self.encoder,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        outputs = self.encoder(input_ids, **kwargs)
        logits = outputs.logits

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
    parser = HfArgumentParser((DaptArguments, TrainingArguments))
    dapt_args: DaptArguments
    training_args: TrainingArguments

    if json_file is not None and json_obj is not None:
        raise ValueError("cannot specify json_file and json_obj")

    if json_file is not None:
        (dapt_args, training_args) = parser.parse_json_file(json_file=json_file)
    elif json_obj is not None:
        (dapt_args, training_args) = parser.parse_dict(json_obj)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        (dapt_args, training_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (dapt_args, training_args) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
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
    logger.info(f"Training arguments {training_args}")

    # Set seed
    set_seed(training_args.seed)

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

    # model = AutoModelForMaskedLM.from_pretrained(dapt_args.encoder_name)
    config = AutoConfig.from_pretrained(dapt_args.encoder_name)
    model = DaptModel(config)

    dataset = DaptDataset(dapt_args, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
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

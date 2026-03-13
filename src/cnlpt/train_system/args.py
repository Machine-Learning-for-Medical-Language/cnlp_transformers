from dataclasses import dataclass, field
from typing import Union

import torch
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments


@dataclass
class CnlpTrainingArguments(TrainingArguments):
    def __post_init__(self):
        if self.metric_for_best_model is None:
            self.metric_for_best_model = "eval_avg_macro_f1"
        elif not self.metric_for_best_model.startswith("eval_"):
            self.metric_for_best_model = f"eval_{self.metric_for_best_model}"

        # `dataloader_pin_memory` is unsupported on mps but defaults to True,
        # so we'll disable it here to avoid warnings in the console.
        if self.dataloader_pin_memory and self.device == torch.device("mps"):
            self.dataloader_pin_memory = False

        return super().__post_init__()

    weight_classes: bool = field(
        default=False,
        metadata={
            "help": "A flag that indicates whether class-specific loss should be used. This can be useful in cases with severe class imbalance. The formula for a weight of a class is the count of that class divided the count of the rarest class."
        },
    )
    final_task_weight: float = field(
        default=1.0,
        metadata={
            "help": "Amount to up/down-weight final task in task list (other tasks weighted 1.0)."
        },
    )
    freeze_encoder: float = field(
        default=0.0,
        metadata={
            "help": "Freeze the encoder layers and only train the layer between the encoder and classification architecture. Probably works best with --token flag since [CLS] may not be well-trained for anything in particular. If not specified, no weight freezing will be done. If specified as a flag (no arguments), 100%% of weights will be frozen. If a float (0..1.0) is specified, each weight will be frozen with that probability.",
            "nargs": "?",
            "const": 1.0,
        },
    )
    bias_fit: bool = field(
        default=False,
        metadata={
            "help": "Only optimize the bias parameters of the encoder (and the weights of the classifier heads), as proposed in the BitFit paper by Ben Zaken et al. 2021 (https://arxiv.org/abs/2106.10199)."
        },
    )
    evals_per_epoch: int = field(
        default=0,
        metadata={
            "help": "Number of times to evaluate and possibly save model per training epoch (allows for a lazy kind of early stopping). Note that setting this argument will automatically override `eval_steps` and `eval_strategy`."
        },
    )

    rich_display: bool = field(
        default=True,
        metadata={
            "help": "Whether to render a live progress display in the console during training."
        },
    )

    # override transformers TrainingArguments defaults
    logging_strategy: IntervalStrategy = field(
        default=IntervalStrategy.EPOCH,
        metadata={"help": "The logging strategy to adopt during training."},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to log the first step of training."},
    )
    cache_dir: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Optionally override the HuggingFace cache directory.",
        },
    )
    metric_for_best_model: Union[str, None] = field(
        default="avg_macro_f1",
        metadata={
            "help": 'The metric to use to compare two different models. Average across tasks with "avg_[acc|macro_f1|micro_f1]". Optimize for a specific task with "taskname.[acc|macro_f1|micro_f1]". Optimize for a particular label with "taskname.labelname.f1". Average multiple metrics with "METRIC_1,METRIC_2".'
        },
    )

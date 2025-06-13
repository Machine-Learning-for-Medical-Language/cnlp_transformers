from dataclasses import dataclass, field

from transformers.training_args import TrainingArguments


@dataclass
class CnlpTrainingArguments(TrainingArguments):
    """
    Additional arguments specific to this class.
    See all possible arguments in :class:`transformers.TrainingArguments`
    or by passing the ``--help`` flag to this script.
    """

    evals_per_epoch: int = field(
        default=-1,
        metadata={
            "help": "Number of times to evaluate and possibly save model per training epoch (allows for a lazy kind of early stopping)"
        },
    )
    final_task_weight: float = field(
        default=1.0,
        metadata={
            "help": "Amount to up/down-weight final task in task list (other tasks weighted 1.0)"
        },
    )
    freeze: float = field(
        default=-1.0,
        metadata={
            "help": "Freeze the encoder layers and only train the layer between the encoder and classification architecture. Probably works best with --token flag since [CLS] may not be well-trained for anything in particular. If not specified, no weight freezing will be done. If specified as a flag (no arguments), 100%% of weights will be frozen. If a float (0..1.0) is specified, each weight will be frozen with that probability.",
            "nargs": "?",
            "const": 1.0,
        },
    )
    bias_fit: bool = field(
        default=False,
        metadata={
            "help": "Only optimize the bias parameters of the encoder (and the weights of the classifier heads), as proposed in the BitFit paper by Ben Zaken et al. 2021 (https://arxiv.org/abs/2106.10199)"
        },
    )
    output_prob: bool = field(
        default=False,
        metadata={
            "help": "If selected, probability scores will be added to the output prediction file for test data when used with --do_predict, and to the evaluation file for dev data when used with --error_analysis.  Currently implemented for classification tasks only."
        },
    )
    truncation_side_left: bool = field(
        default=False,
        metadata={
            "help": "Truncate samples from left. Note that hier model do not support this setting."
        },
    )

    error_analysis: bool = field(
        default=False,
        metadata={
            "help": "Pretty printing for instances where at least one ground truth label for any of the tasks disagrees with the model's prediction"
        },
    )

    logging_strategy: str = field(default="epoch")

    rich_display: bool = field(
        default=True,
        metadata={
            "help": "Whether to render a live progress display in the console during training."
        },
    )

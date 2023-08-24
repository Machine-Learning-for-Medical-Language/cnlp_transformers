"""
Module containing the CNLP command line argument definitions
"""

from typing import Callable, Dict, Optional, List, Union, Any
from dataclasses import dataclass, field, fields
from enum import Enum
from transformers import TrainingArguments


@dataclass
class CnlpTrainingArguments(TrainingArguments):
    """
    Additional arguments specific to this class.
    See all possible arguments in :class:`transformers.TrainingArguments`
    or by passing the ``--help`` flag to this script.
    """

    evals_per_epoch: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of times to evaluate and possibly save model per training epoch (allows for a lazy kind of early stopping)"
        },
    )
    final_task_weight: Optional[float] = field(
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


cnlpt_models = ["cnn", "lstm", "hier", "cnlpt"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    See all possible arguments by passing the ``--help`` flag to this script.
    """

    model: Optional[str] = field(
        default="cnlpt", metadata={"help": "Model type", "choices": cnlpt_models}
    )
    encoder_name: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    layer: Optional[int] = field(
        default=-1, metadata={"help": "Which layer's CLS ('<s>') token to use"}
    )
    token: bool = field(
        default=False,
        metadata={
            "help": "Classify over an actual token rather than the [CLS] ('<s>') token -- requires that the tokens to be classified are surrounded by <e>/</e> tokens"
        },
    )

    # NxN relation classifier-specific arguments
    num_rel_feats: Optional[int] = field(
        default=12,
        metadata={
            "help": "Number of features/attention heads to use in the NxN relation classifier"
        },
    )
    head_features: Optional[int] = field(
        default=64,
        metadata={
            "help": "Number of parameters in each attention head in the NxN relation classifier"
        },
    )

    # CNN-specific arguments
    cnn_embed_dim: Optional[int] = field(
        default=100,
        metadata={
            "help": "For the CNN baseline model, the size of the word embedding space."
        },
    )
    cnn_num_filters: Optional[int] = field(
        default=25,
        metadata={
            "help": (
                "For the CNN baseline model, the number of "
                "convolution filters to use for each filter size."
            )
        },
    )

    cnn_filter_sizes: Optional[List[int]] = field(
        default_factory=lambda: [1, 2, 3],
        metadata={
            "help": (
                "For the CNN baseline model, a space-separated list "
                "of size(s) of the filters (kernels)"
            )
        },
    )

    # LSTM-specific arguments
    lstm_embed_dim: Optional[int] = field(
        default=100,
        metadata={
            "help": "For the LSTM baseline model, the size of the word embedding space."
        },
    )
    lstm_hidden_size: Optional[int] = field(
        default=100,
        metadata={
            "help": "For the LSTM baseline model, the hidden size of the LSTM layer"
        },
    )

    # Multi-task classifier-specific arguments
    use_prior_tasks: bool = field(
        default=False,
        metadata={
            "help": "In the multi-task setting, incorporate the logits from the previous tasks into subsequent representation layers. This will be done in the task order specified in the command line."
        },
    )

    # Hierarchical Transformer-specific arguments
    hier_num_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "For the hierarchical model, the number of document-level transformer "
                "layers"
            )
        },
    )
    hier_hidden_dim: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "For the hierarchical model, the inner hidden size of the positionwise "
                "FFN in the document-level transformer layers"
            )
        },
    )
    hier_n_head: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the number of attention heads in the "
                "document-level transformer layers"
            )
        },
    )
    hier_d_k: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the size of the query and key vectors in "
                "the document-level transformer layers"
            )
        },
    )
    hier_d_v: Optional[int] = field(
        default=96,
        metadata={
            "help": (
                "For the hierarchical model, the size of the value vectors in the "
                "document-level transformer layers"
            )
        },
    )
    hier_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "For the hierarchical model, the dropout probability for the "
            "document-level transformer layers"
        },
    )
    keep_existing_classifiers: bool = field(
        default=False,
        metadata={
            "help": (
                "For the hierarchical model, load classifier weights from "
                "the saved checkpoint. For inference of the trained model or "
                "continued fine-tuning."
            )
        },
    )
    ignore_existing_classifiers: bool = field(
        default=False,
        metadata={
            "help": (
                "For the hierarchical model, ignore classifier weights "
                "from the saved checkpoint. The weights will be initialized."
            )
        },
    )

    def to_dict(self):
        # adapted from transformers.TrainingArguments.to_dict()
        # filter out fields that are defined as field(init=False)
        d = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.init
        }

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class DaptArguments:
    encoder_name: Optional[str] = field(
        default="roberta-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    output_dir: Optional[str] = field(
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
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data dir for domain-adaptive pretraining."}
    )
    cache_dir: Optional[str] = field(
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

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Union

cnlpt_models = ["cnn", "lstm", "hier", "cnlpt"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    See all possible arguments by passing the ``--help`` flag to this script.
    """

    model: Union[str, None] = field(
        default="cnlpt", metadata={"help": "Model type", "choices": cnlpt_models}
    )
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
    cache_dir: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    layer: int = field(
        default=-1, metadata={"help": "Which layer's CLS ('<s>') token to use"}
    )
    token: bool = field(
        default=False,
        metadata={
            "help": "Classify over an actual token rather than the [CLS] ('<s>') token -- requires that the tokens to be classified are surrounded by <e>/</e> tokens"
        },
    )

    # NxN relation classifier-specific arguments
    num_rel_feats: int = field(
        default=12,
        metadata={
            "help": "Number of features/attention heads to use in the NxN relation classifier"
        },
    )
    head_features: int = field(
        default=64,
        metadata={
            "help": "Number of parameters in each attention head in the NxN relation classifier"
        },
    )

    # CNN-specific arguments
    cnn_embed_dim: int = field(
        default=100,
        metadata={
            "help": "For the CNN baseline model, the size of the word embedding space."
        },
    )
    cnn_num_filters: int = field(
        default=25,
        metadata={
            "help": (
                "For the CNN baseline model, the number of "
                "convolution filters to use for each filter size."
            )
        },
    )

    cnn_filter_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 3],
        metadata={
            "help": (
                "For the CNN baseline model, a space-separated list "
                "of size(s) of the filters (kernels)"
            )
        },
    )

    # LSTM-specific arguments
    lstm_embed_dim: int = field(
        default=100,
        metadata={
            "help": "For the LSTM baseline model, the size of the word embedding space."
        },
    )
    lstm_hidden_size: int = field(
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
    hier_num_layers: int = field(
        default=2,
        metadata={
            "help": (
                "For the hierarchical model, the number of document-level transformer "
                "layers"
            )
        },
    )
    hier_hidden_dim: int = field(
        default=2048,
        metadata={
            "help": (
                "For the hierarchical model, the inner hidden size of the positionwise "
                "FFN in the document-level transformer layers"
            )
        },
    )
    hier_n_head: int = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the number of attention heads in the "
                "document-level transformer layers"
            )
        },
    )
    hier_d_k: int = field(
        default=8,
        metadata={
            "help": (
                "For the hierarchical model, the size of the query and key vectors in "
                "the document-level transformer layers"
            )
        },
    )
    hier_d_v: int = field(
        default=96,
        metadata={
            "help": (
                "For the hierarchical model, the size of the value vectors in the "
                "document-level transformer layers"
            )
        },
    )
    hier_dropout: float = field(
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

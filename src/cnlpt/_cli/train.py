from enum import Enum
from typing import Annotated, Any, Final, Union

import typer
from click.core import ParameterSource
from transformers.hf_argparser import HfArgumentParser
from transformers.models.auto.modeling_auto import AutoModel
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from ..data.cnlp_dataset import CnlpDataset, HierarchicalDataConfig, TruncationSide
from ..modeling.config.cnn_config import CnnModelConfig
from ..modeling.config.hierarchical_config import HierarchicalModelConfig
from ..modeling.config.lstm_config import LstmModelConfig
from ..modeling.config.projection_config import ProjectionModelConfig
from ..modeling.models import CnnModel, HierarchicalModel, LstmModel, ProjectionModel
from ..modeling.types import ClassificationMode, ModelType
from ..train_system.args import CnlpTrainingArguments
from ..train_system.cnlp_train_system import CnlpTrainSystem

_ARG_COMPAT_METADATA_KEY = "cnlpt.model_arg_compat"
DEFAULT_ENCODER: Final = "roberta-base"


def compatible_models(types: list[ModelType]):
    def callback(ctx: typer.Context, param: typer.CallbackParam, value: Any):
        if ctx.resilient_parsing:
            return
        if _ARG_COMPAT_METADATA_KEY not in ctx.meta:
            ctx.meta[_ARG_COMPAT_METADATA_KEY] = {}
        ctx.meta[_ARG_COMPAT_METADATA_KEY][param.name] = types
        if isinstance(value, Enum):
            return value.value
        return value

    return callback


def training_arg_option(
    field_name: str,
    *aliases,
    compatibility: Union[list[ModelType], None] = None,
    **kwargs,
):
    field = CnlpTrainingArguments.__dataclass_fields__[field_name]
    if len(aliases) == 0:
        aliases = (f"--{field_name}",)
    if compatibility is not None:
        kwargs["callback"] = compatible_models(compatibility)
    return typer.Option(
        *aliases,
        help=field.metadata.get("help", None),
        rich_help_panel="CNLPT Training Arguments",
        **kwargs,
    )


def model_arg_option(
    *args,
    compatibility: Union[list[ModelType], None] = None,
    **kwargs,
):
    if compatibility is not None:
        kwargs["callback"] = compatible_models(compatibility)
    return typer.Option(*args, rich_help_panel="Model Arguments", **kwargs)


def data_arg_option(
    *args,
    compatibility: Union[list[ModelType], None] = None,
    **kwargs,
):
    if compatibility is not None:
        kwargs["callback"] = compatible_models(compatibility)
    return typer.Option(*args, rich_help_panel="Data Arguments", **kwargs)


##### MODEL ARGS #####
ModelTypeArg = Annotated[
    ModelType,
    model_arg_option(
        "--model_type",
        help="The type of model to load.",
        case_sensitive=False,
    ),
]
EncoderArg = Annotated[
    str,
    model_arg_option(
        "--encoder",
        compatibility=["proj", "hier"],
        help="For projection and hierarchical models, which encoder model to use.",
    ),
]
UsePriorTasksArg = Annotated[
    bool,
    model_arg_option(
        "--use_prior_tasks",
        compatibility=["proj", "cnn"],
        help="For projection and CNN models, whether to use the output of prior tasks as an input to subsequent ones.",
    ),
]
EncoderLayerArg = Annotated[
    int,
    model_arg_option(
        "--encoder_layer",
        "--layer",
        compatibility=["proj", "hier"],
        help="For projection and hierarchical models, which layer of the encoder to use for representation.",
    ),
]
ClassificationModeArg = Annotated[
    ClassificationMode,
    model_arg_option(
        "--classification_mode",
        compatibility=["proj"],
        help="For projection models, chooses whether to classify from the [CLS] token or from a token span tagged with <e></e>.",
        case_sensitive=False,
    ),
]
RelationAttnHeadsArg = Annotated[
    int,
    model_arg_option(
        "--relation_attn_heads",
        compatibility=["proj"],
        help="For projection models, the number of relation attention heads to use for relation extraction tasks.",
    ),
]
RelationAttnHeadDimArg = Annotated[
    int,
    model_arg_option(
        "--relation_attn_head_dim",
        compatibility=["proj"],
        help="For projection models, the dimension of attention heads for relation extraction tasks.",
    ),
]
HierLayersArg = Annotated[
    int,
    model_arg_option(
        "--hier_layers",
        compatibility=["hier"],
        help="For hierarchical models, the number of hierarchical layers.",
    ),
]
HierUseLayerArg = Annotated[
    int,
    model_arg_option(
        "--hier_layers",
        compatibility=["hier"],
        help="For hierarchical models, the layer to use for classification.",
    ),
]
HierHiddenDimArg = Annotated[
    int,
    model_arg_option(
        "--hier_hidden_dim",
        compatibility=["hier"],
        help="For hierarchical models, the hidden dimension of the FFN in each layer.",
    ),
]
HierHeadsArg = Annotated[
    int,
    model_arg_option(
        "--hier_heads",
        compatibility=["hier"],
        help="For hierarchical models, the number of attention heads.",
    ),
]
HierQKDimArg = Annotated[
    int,
    model_arg_option(
        "--hier_qk_dim",
        compatibility=["hier"],
        help="For hierarchical models, the dimension of the query and key vectors.",
    ),
]
HierVDimArg = Annotated[
    int,
    model_arg_option(
        "--hier_v_dim",
        compatibility=["hier"],
        help="For hierarchical models, the dimension of the value vectors.",
    ),
]
DropoutArg = Annotated[
    # TODO(ian): should this be available for proj models too?
    float,
    model_arg_option(
        "--dropout",
        compatibility=["hier", "cnn", "lstm"],
        help="For hierarchical, CNN, and LSTM models, the dropout probability.",
    ),
]
EmbedDimArg = Annotated[
    int,
    model_arg_option(
        "--embed_dim",
        compatibility=["cnn", "lstm"],
        help="For CNN and LSTM models, the embedding dimension.",
    ),
]
CnnNumFiltersArg = Annotated[
    int,
    model_arg_option(
        "--cnn_num_filters",
        compatibility=["cnn"],
        help="For CNN models, the number of filters per filter size.",
    ),
]
CnnFilterSizesArg = Annotated[
    str,
    model_arg_option(
        "--cnn_filter_sizes",
        compatibility=["cnn"],
        help="For CNN models, a comma-separated list of filter sizes to use.",
    ),
]
LstmHiddenSizeArg = Annotated[
    int,
    model_arg_option(
        "--lstm_hidden_size",
        compatibility=["lstm"],
        help="LSTM models, the dimension of the hidden layer.",
    ),
]

##### DATA ARGS #####
DataDirArg = Annotated[
    str,
    data_arg_option(
        "--data_dir", help="Path to a directory containing CNLPT-formatted data."
    ),
]
TaskNamesArg = Annotated[
    Union[list[str], None],
    data_arg_option(
        "--task",
        "-t",
        help="The name of a task in the dataset to train on. Can be specified multiple times to target more than one task. Defaults to all tasks.",
    ),
]
TokenizerArg = Annotated[
    Union[str, None],
    data_arg_option(
        "--tokenizer",
        help=f'Name or path to a model to use for tokenization. For projection and hierarchical models, this will default to the --encoder_name if left unspecified; otherwise defaults to "{DEFAULT_ENCODER}".',
    ),
]
TruncationSideArg = Annotated[
    TruncationSide,
    data_arg_option(
        "--truncation_side",
        help="Which side to perform truncation when tokenizing. Note that hierarchical models don't support left-side truncation.",
        compatibility=["cnn", "lstm", "proj"],
        case_sensitive=False,
    ),
]
MaxSeqLengthArg = Annotated[
    int,
    data_arg_option(
        "--max_seq_length", help="Maximum sequence length for tokenization."
    ),
]
OverwriteDataCacheArg = Annotated[
    bool,
    data_arg_option(
        "--overwrite_data_cache",
        help="Overwrite the data cache to force re-preprocessing of the data.",
    ),
]
MaxTrainArg = Annotated[
    Union[int, None],
    data_arg_option("--max_train", help="Limit the number of training samples to use."),
]
MaxEvalArg = Annotated[
    Union[int, None],
    data_arg_option("--max_eval", help="Limit the number of eval samples to use."),
]
MaxTestArg = Annotated[
    Union[int, None],
    data_arg_option("--max_test", help="Limit the number of test samples to use."),
]
AllowDisjointLabelsArg = Annotated[
    bool,
    data_arg_option(
        "--allow_disjoint_labels",
        help="Allow disjoint label sets for tasks in different data splits. Can be useful for debugging.",
    ),
]
CharacterLevelArg = Annotated[
    bool,
    data_arg_option(
        "--character_level",
        help=".Whether the dataset sould be processed at the character level (otherwise will be processed at the token level).",
    ),
]
HierChunkLenArg = Annotated[
    Union[int, None],
    data_arg_option("--hier_chunk_len", help="Chunk length for hierarchical models."),
]
HierNumChunksArg = Annotated[
    Union[int, None],
    data_arg_option(
        "--hier_num_chunks", help="Number of chunks for hierarchical models."
    ),
]
HierPrependEmptyChunkArg = Annotated[
    Union[int, None],
    data_arg_option(
        "--hier_prepend_empty_chunk",
        help="Whether to prepend an empty chunk for hierarchical models.",
    ),
]

##### TRAINING ARGS #####
WeightClassesArg = Annotated[bool, training_arg_option("weight_classes")]
FinalTaskWeightArg = Annotated[float, training_arg_option("final_task_weight")]
FreezeEncoderArg = Annotated[float, training_arg_option("freeze_encoder")]
BiasFitArg = Annotated[bool, training_arg_option("bias_fit")]
ReportProbsArg = Annotated[bool, training_arg_option("report_probs")]
EvalsPerEpochArg = Annotated[int, training_arg_option("evals_per_epoch")]
RichDisplayArg = Annotated[bool, training_arg_option("rich_display")]
LoggingStrategyArg = Annotated[
    IntervalStrategy, training_arg_option("logging_strategy")
]
LoggingFirstStepArg = Annotated[bool, training_arg_option("logging_first_step")]
CacheDirArg = Annotated[Union[str, None], training_arg_option("cache_dir")]


def train(
    ctx: typer.Context,
    # ------------------ #
    #     MODEL ARGS     #
    # ------------------ #
    model_type: ModelTypeArg = ...,
    encoder_name: EncoderArg = DEFAULT_ENCODER,
    use_prior_tasks: UsePriorTasksArg = False,
    encoder_layer: EncoderLayerArg = -1,
    classification_mode: ClassificationModeArg = "cls",
    relation_attn_heads: RelationAttnHeadsArg = 12,
    relation_attn_head_dim: RelationAttnHeadDimArg = 64,
    hier_layers: HierLayersArg = 8,
    hier_use_layer: HierUseLayerArg = -1,
    hier_hidden_dim: HierHiddenDimArg = 2048,
    hier_heads: HierHeadsArg = 8,
    hier_qk_dim: HierQKDimArg = 8,
    hier_v_dim: HierVDimArg = 96,
    dropout: DropoutArg = 0.1,
    embed_dim: EmbedDimArg = 100,
    cnn_num_filters: CnnNumFiltersArg = 25,
    cnn_filter_sizes: CnnFilterSizesArg = "1,2,3",
    lstm_hidden_size: LstmHiddenSizeArg = 100,
    # ----------------- #
    #     DATA ARGS     #
    # ----------------- #
    data_dir: DataDirArg = ...,
    task_names: TaskNamesArg = None,
    tokenizer: TokenizerArg = DEFAULT_ENCODER,
    truncation_side: TruncationSideArg = "right",
    max_seq_length: MaxSeqLengthArg = 128,
    overwrite_data_cache: OverwriteDataCacheArg = False,
    max_train: MaxTrainArg = None,
    max_eval: MaxEvalArg = None,
    max_test: MaxTestArg = None,
    allow_disjoint_labels: AllowDisjointLabelsArg = False,
    character_level: CharacterLevelArg = False,
    hier_chunk_len: HierChunkLenArg = None,
    hier_num_chunks: HierNumChunksArg = None,
    hier_prepend_empty_chunk: HierPrependEmptyChunkArg = None,
    # --------------------- #
    #     TRAINING ARGS     #
    # --------------------- #
    weight_classes: WeightClassesArg = False,
    final_task_weight: FinalTaskWeightArg = 1.0,
    freeze_encoder: FreezeEncoderArg = 0.0,
    bias_fit: BiasFitArg = False,
    report_probs: ReportProbsArg = False,
    evals_per_epoch: EvalsPerEpochArg = 0,
    rich_display: RichDisplayArg = True,
    logging_strategy: LoggingStrategyArg = "epoch",
    logging_first_step: LoggingFirstStepArg = True,
    cache_dir: CacheDirArg = None,
    # --------------------- #
    **kwargs,
):
    # TODO(ian): it's probably worth making this docstring pretty descriptive
    """Run the cnlp_transformers training system."""

    # If the tokenizer wasn't explicitly specified and this is a model
    # that accepts an encoder, use the encoder's tokenizer.
    if ctx.get_parameter_source("tokenizer") != ParameterSource.COMMANDLINE and (
        model_type in (ModelType.HIER, ModelType.PROJ)
    ):
        tokenizer = encoder_name

    dataset = CnlpDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        task_names=task_names,
        hier_config=(
            HierarchicalDataConfig(
                hier_chunk_len, hier_num_chunks, hier_prepend_empty_chunk
            )
            if model_type == ModelType.HIER
            else None
        ),
        truncation_side=truncation_side,
        max_seq_length=max_seq_length,
        use_data_cache=not overwrite_data_cache,
        max_train=max_train,
        max_eval=max_eval,
        max_test=max_test,
        allow_disjoint_labels=allow_disjoint_labels,
        character_level=character_level,
        hf_cache_dir=cache_dir,
    )

    # Since CnlpTrainingArguments inherits from the transformers TrainingArguments,
    # rather than maintain explicit arguments for all TrainingArguments fields we'll
    # just leave them unprocessed and pass any unknown args to the transformers parser.
    hf_args_parser = HfArgumentParser(TrainingArguments)
    hf_training_args, extra_args = hf_args_parser.parse_known_args(ctx.args)

    if len(extra_args) > 0:
        raise typer.BadParameter(f"unrecognized arguments: {extra_args!s}", ctx)

    # Combine our args with the args parsed by transformers. We must use the `|`
    # operator in this order so that our args take precedence.
    training_args = CnlpTrainingArguments(
        **(
            vars(hf_training_args)
            | dict(
                weight_classes=weight_classes,
                final_task_weight=final_task_weight,
                freeze_encoder=freeze_encoder,
                bias_fit=bias_fit,
                report_probs=report_probs,
                evals_per_epoch=evals_per_epoch,
                rich_display=rich_display,
                logging_strategy=logging_strategy,
                logging_first_step=logging_first_step,
            )
        )
    )

    if model_type == ModelType.CNN:
        config = CnnModelConfig(
            tasks=list(dataset.tasks),
            vocab_size=len(dataset.tokenizer),
            use_prior_tasks=use_prior_tasks,
            embed_dim=embed_dim,
            num_filters=cnn_num_filters,
            filter_sizes=tuple([int(s.strip()) for s in cnn_filter_sizes.split(",")]),
            dropout=dropout,
        )
    elif model_type == ModelType.LSTM:
        config = LstmModelConfig(
            tasks=list(dataset.tasks),
            vocab_size=len(dataset.tokenizer),
            embed_dim=embed_dim,
            hidden_size=lstm_hidden_size,
            dropout=dropout,
        )
    elif model_type == ModelType.HIER:
        config = HierarchicalModelConfig(
            tasks=list(dataset.tasks),
            vocab_size=len(dataset.tokenizer),
            encoder_name=encoder_name,
            layer=hier_use_layer,
            n_layers=hier_layers,
            d_inner=hier_hidden_dim,
            n_head=hier_heads,
            d_k=hier_qk_dim,
            d_v=hier_v_dim,
            dropout=dropout,
        )
    elif model_type == ModelType.PROJ:
        config = ProjectionModelConfig(
            tasks=list(dataset.tasks),
            vocab_size=len(dataset.tokenizer),
            encoder_name=encoder_name,
            encoder_layer=encoder_layer,
            use_prior_tasks=use_prior_tasks,
            classification_mode=classification_mode,
            num_rel_attention_heads=relation_attn_heads,
            rel_attention_head_dims=relation_attn_head_dim,
            character_level=character_level,
        )

    model_init_kwargs = {}
    if weight_classes:
        model_init_kwargs["class_weights"] = dataset.get_class_weights(
            training_args.device
        )
    if freeze_encoder > 0:
        model_init_kwargs["freeze"] = freeze_encoder
    if final_task_weight != 1.0:
        model_init_kwargs["final_task_weight"] = final_task_weight
    if bias_fit:
        model_init_kwargs["bias_fit"] = True

    model: Union[CnnModel, LstmModel, HierarchicalModel, ProjectionModel] = (
        AutoModel.from_config(config, **model_init_kwargs)
    )
    train_system = CnlpTrainSystem(model, dataset, training_args)
    train_system.train()


TRAIN_EPILOG = """[red]More training arguments are available, see the
[b blue][link=https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments]HF Transformers documentation[/link][/b blue]."""

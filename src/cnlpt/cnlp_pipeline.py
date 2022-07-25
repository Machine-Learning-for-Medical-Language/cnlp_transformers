import os
import sys
import numpy as np

from dataclasses import dataclass, field

from .cnlp_pipeline_utils import (
    model_dicts,
    get_sentences_and_labels,
    get_predictions,
)

from .cnlp_processors import classifier_to_relex, cnlp_compute_metrics

from .CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

from transformers import AutoConfig, AutoModel, HfArgumentParser

SPECIAL_TOKENS = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']

modes = ["inf", "eval"]


@dataclass
class PipelineArguments:
    """
    Arguments pertaining to the models, mode, and data used by the pipeline.
    """
    models_dir: str = field(
        metadata={
            "help": (
                "Path where each entity model is stored "
                "in a folder named after its "
                "corresponding cnlp_processor, "
                "models with a 'tagging' output mode will be run first "
                "followed by models with a 'classification' "
                "ouput mode over the assembled data"
            )
        }
    )
    in_file: str = field(
        metadata={
            "help": (
                "Path to file, with one raw sentence"
                "per line in the case of inference,"
                " and one <label>\t<annotated sentence> "
                "per line in the case of evaluation"
            )
        }
    )
    mode: str = field(
        default="inf",
        metadata={
            "help": (
                "Use mode for full pipeline, "
                "inference, which outputs annotated sentences "
                "and their relation, or eval, "
                "which outputs metrics for a provided set of samples "
                "(requires labels)"
            )
        }
    )
    axis_task: str = field(
        default="dphe_med",
        metadata={
            "help": (
                "key of the task in cnlp_processors "
                "which generates the tag that will map to <a1> <mention> </a1>"
                " in pairwise annotations"
            )
        }
    )


def main():
    parser = HfArgumentParser(PipelineArguments)

    if (
            len(sys.argv) == 2
            and sys.argv[1].endswith(".json")
    ):
        # If we pass only one argument to the script
        # and it's the path to a json file,
        # let's parse it to get our arguments.

        # the ',' is to deal with unary tuple weirdness
        pipeline_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        pipeline_args, = parser.parse_args_into_dataclasses()

    if pipeline_args.mode == "inf":
        inference(pipeline_args)
    elif pipeline_args.mode == "eval":
        evaluation(pipeline_args)
    else:
        ValueError("Invalid pipe mode!")


def inference(pipeline_args):
    # Required for loading cnlpt models using Huggingface
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(pipeline_args.models_dir)

    # Only need raw sentences for inference
    _, sentences, _ = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="inf",
        task_names=out_model_dict.keys(),
    )

    # Inference mode takes care of
    # printing, don't need the predictions
    # dictionary or dict->matrix function
    _, _, _, _ = get_predictions(
        sentences,
        taggers_dict,
        out_model_dict,
        pipeline_args.axis_task,
        mode='inf',
    )


def evaluation(pipeline_args):
    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    taggers_dict, out_model_dict = model_dicts(pipeline_args.models_dir)

    # For eval need ground truth
    # labels as well as the length of
    # the longest sentence in the split
    # for matrix generation and padding
    (
        idx_labels_dict,
        sentences,
        split_max_len,
    ) = get_sentences_and_labels(
        in_file=pipeline_args.in_file,
        mode="eval",
        task_names=out_model_dict.keys(),
    )

    predictions_dict, local_relex, _, _ = get_predictions(
        sentences,
        taggers_dict,
        out_model_dict,
        pipeline_args.axis_task,
        mode='eval',
    )

    for task_name, prediction_tuples in predictions_dict.items():
        report = cnlp_compute_metrics(
            classifier_to_relex[task_name],
            # Giant relex matrix of the predictions
            np.array(
                [local_relex(sent_preds, split_max_len) for
                 sent_preds in prediction_tuples]
            ),
            # Giant relex matrix of the ground
            # truth labels
            np.array(
                [local_relex(sent_labels, split_max_len) for
                 sent_labels in idx_labels_dict[task_name]]
            )
        )
        print(report)


if __name__ == "__main__":
    main()

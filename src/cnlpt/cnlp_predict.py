import numpy as np
import pandas as pd
import re
import csv

from datasets import Dataset
from transformers.trainer_utils import EvalPrediction
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from typing import Dict, List, Tuple, Union
from itertools import chain
from operator import itemgetter
from collections import defaultdict


def remove_newline(review):
    review = review.replace("&#039;", "'")
    review = review.replace("\n", " <cr> ")
    review = review.replace("\r", " <cr> ")
    review = review.replace("\t", " ")
    return review


def compute_disagreements(
    preds: np.ndarray,
    labels: np.ndarray,
    output_mode: str,
) -> np.ndarray:
    """
    Function that defines and computes the metrics used for each task.
    When adding a task definition to this file, add a branch to this
    function defining what its evaluation metric invocation should be.
    If the new task is a simple classification task, a sensible default
    is defined; falling back on this will trigger a warning.

    :param str task_name: the task name used to index into cnlp_processors
    :param numpy.ndarray preds: the predicted labels from the model
    :param numpy.ndarray labels: the true labels
    :rtype: typing.Dict[str, typing.Any]
    :return: a dictionary containing evaluation metrics
    """

    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if output_mode == classification:
        return classification_disagreements(preds=preds, labels=labels)
    elif output_mode == tagging:
        return tagging_disagreements(preds=preds, labels=labels)
    elif output_mode == relex:
        return relation_disagreements(
            preds=preds,
            labels=labels,
        )
    else:
        raise Exception("As yet unsupported task in cnlpt")


def classification_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where(np.not_equal(preds, labels))
    return indices


def tagging_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where([*map(any, np.not_equal(preds, labels))])
    return indices


def relation_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where([*map(lambda s: s.any(), np.not_equal(preds, labels))])
    return indices


def process_prediction(
    task_names: List[str],
    output_fn: str,
    error_analysis: bool,
    task_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray]],
    task_to_label_boundaries: Dict[str, Tuple[int, int]],
    eval_dataset,
    task2labels: Dict[str, List[str]],
    # split_name: str,
    # dataset_ind: int,
    # dataset: ClinicalNlpDataset,
    output_mode: Dict[str, str],
):
    task_to_error_inds: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
    if error_analysis:
        for task, label_packet in task_to_label_packet.items():
            preds, labels = label_packet
            task_to_error_inds[task] = compute_disagreements(
                preds, labels, output_mode[task]
            )

    # start_ind = end_ind = 0
    # for ind in range(dataset_ind):
    #     start_ind += len(dataset.datasets[ind][split_name])
    # end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    # eval_dataset = Dataset.from_dict(
    #     dataset.processed_dataset[split_name][start_ind:end_ind]
    # )

    if error_analysis:
        relevant_indices = set(chain.from_iterable(task_to_error_inds.values()))

    else:
        relevant_indices = set(range(len(eval_dataset["text"])))

    out_table = pd.DataFrame(
        columns=["text", *sorted(task_names)],
        index=relevant_indices,
    )

    out_table["text"] = [eval_dataset["text"][index] for index in relevant_indices]
    out_table["text"] = out_table["text"].apply(remove_newline)

    out_table["text"] = out_table["text"].str.replace('"', "")
    out_table["text"] = out_table["text"].str.replace("//", "")
    out_table["text"] = out_table["text"].str.replace("\\", "")
    torch_labels = np.array(eval_dataset["label"])
    # task2labels = dataset.get_labels()
    # for task_label, error_inds in task_to_error_inds.items():
    for task_name, packet in task_to_label_packet.items():
        preds, labels = packet
        task_labels = task2labels[task_name]
        if error_analysis:
            error_inds = task_to_error_inds[task_name]
            out_table[task_name][error_inds] = get_output_list(
                error_analysis,
                task_name,
                task_labels,
                task_to_label_boundaries,
                preds,
                labels,
                output_mode,
                error_inds,
                torch_labels,
            )
        else:
            out_table[task_name] = get_output_list(
                error_analysis,
                task_name,
                task_labels,
                task_to_label_boundaries,
                preds,
                labels,
                output_mode,
                None,
                torch_labels,
            )
    out_table.to_csv(
        output_fn,
        sep="\t",
        index=True,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


# might be more efficient to return a pd.Series or something for the
# assignment and populate it via a generator but for now just use a list
def get_output_list(
    error_analysis: bool,
    pred_task: str,
    task_labels: List[str],
    task2boundaries: Dict[str, Tuple[int, int]],
    prediction: np.ndarray,
    labels: np.ndarray,
    output_mode: Dict[str, str],
    error_inds: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    if len(error_inds) > 0 and error_analysis:
        ground_truth = labels[
            error_inds
        ]  # np.array(eval_dataset[pred_task])[error_inds]
        task_prediction = prediction[error_inds]  # [task2ind[pred_task]][error_inds]
        all_torch_labels = torch_labels[
            error_inds
        ]  # np.array(eval_dataset["label"])[error_inds]
    else:
        ground_truth = labels  # np.array(eval_dataset[pred_task])
        task_prediction = prediction  # [task2ind[pred_task]]
        all_torch_labels = torch_labels  # np.array(eval_dataset["label"])
    task_type = output_mode[pred_task]
    # task_labels = task2labels[pred_task]
    # get the feeling this doesn't work for multiple tasks but we'll
    # probe those data structures when we run the code
    if task_type == classification:
        return get_classification_prints(task_labels, ground_truth, task_prediction)
    elif task_type == tagging:
        task_torch_labels = all_torch_labels
        if pred_task in task2boundaries.keys():
            labels_start, labels_end = task2boundaries[pred_task]
            task_torch_labels = all_torch_labels[:, :, labels_start:labels_end]
        return get_tagging_prints(
            task_labels, ground_truth, task_prediction, task_torch_labels
        )
    elif task_type == relex:
        task_torch_labels = all_torch_labels
        if pred_task in task2boundaries.keys():
            labels_start, labels_end = task2boundaries[pred_task]
            task_torch_labels = all_torch_labels[:, :, labels_start:labels_end]
        return get_relex_prints(
            task_labels, ground_truth, task_prediction, task_torch_labels
        )
    else:
        return len(error_inds) * ["UNSUPPORTED TASK TYPE"]


def get_classification_prints(
    classification_labels: List[str],
    ground_truths: np.ndarray,  # List[str],
    task_predictions: np.ndarray,
) -> List[str]:
    resolved_predictions = task_predictions  # np.argmax(task_predictions, axis=1)
    predicted_labels = [classification_labels[index] for index in resolved_predictions.astype("int")]

    ground_strings = [classification_labels[index] for index in ground_truths.astype("int")]
    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        return f"Ground: {ground} , Predicted {predicted}"

    return [*map(clean_string, zip(ground_strings, predicted_labels))]


def get_tagging_prints(
    tagging_labels: List[str],
    ground_truths: np.ndarray,
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    resolved_predictions = task_predictions

    def human_readable_labels(tag_token_arrays: Tuple[np.ndarray, np.ndarray]) -> str:
        raw_tags, token_ids = tag_token_arrays
        # since resolved predictions is num errors x seq length
        # and torch labels is num errors x seq length x 1
        return " ".join(
            [
                tagging_labels[label_idx]
                for label_idx in raw_tags[
                    np.where(token_ids.reshape(-1) != -100)
                ].astype("int")
            ]
        )

    # do naive approach for now
    def clean_string(gp_strings: Tuple[str, str]) -> str:
        ground, predicted = gp_strings

        return f"Ground: {ground} , Predicted {predicted}"

    return [
        *map(
            clean_string,
            zip(
                map(human_readable_labels, zip(ground_truths, torch_labels)),
                map(human_readable_labels, zip(resolved_predictions, torch_labels)),
            ),
        )
    ]


def get_relex_prints(
    relex_labels: List[str],
    ground_truths: np.ndarray,  # List[str],
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    resolved_predictions = task_predictions  # np.argmax(task_predictions, axis=3)
    none_index = relex_labels.index("None") if "None" in relex_labels else -1

    def cell2cnlptstr(index_val_pair: Tuple[Tuple[int, ...], int]) -> str:
        index, value = index_val_pair
        first_token, second_token = index
        return f"({first_token}, {second_token}, {relex_labels[value]})"

    def matrix_to_label(matrix: np.ndarray) -> str:
        if all(map(lambda s: s == none_index, matrix.flatten())):
            return "none"

        return " , ".join(
            map(
                cell2cnlptstr,
                filter(lambda s: s[1] == none_index, np.ndenumerate(matrix)),
            )
        )

    def human_readable_labels(index: int) -> str:
        # resolved_predictions[index]  shape is sent length x sent length
        # not the same shape insanity that we had for tagging

        reduced_prediction = np.array(
            [
                *filter(
                    len,
                    [
                        pred_row[np.where(ground_row != -100)]
                        for pred_row, ground_row in zip(
                            resolved_predictions[index], torch_labels[index]
                        )
                    ],
                )
            ]
        )
        return matrix_to_label(reduced_prediction)

    # do naive approach for now
    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        pred_str = " ".join(predicted)

        return f"ground: {ground} , predicted {pred_str}"

    return [
        *map(
            clean_string,
            zip(
                ground_truths,
                map(human_readable_labels, range(resolved_predictions.shape[0])),
            ),
        )
    ]

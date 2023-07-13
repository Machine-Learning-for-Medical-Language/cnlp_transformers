import numpy as np
import pandas as pd

from datasets import Dataset
from transformers.trainer_utils import EvalPrediction
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from .train_system import structure_labels
from typing import Dict, List, Tuple, Union
from itertools import chain
from operator import itemgetter
from collections import defaultdict


def restructure_prediction(
    task_names: List[str],
    raw_prediction: EvalPrediction,
    max_seq_length: int,
    tagger: Dict[str, bool],
    relations: Dict[str, bool],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[int, int]]]:
    task_label_ind = 0

    # disagreement collection stuff for this scope
    task_label_to_boundaries: Dict[str, Tuple[int, int]] = {}
    task_label_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for task_ind, task_name in enumerate(task_names):
        preds, labels, pad = structure_labels(
            raw_prediction,
            task_name,
            task_ind,
            task_label_ind,
            max_seq_length,
            tagger,
            relations,
            task_label_to_boundaries,
        )
        task_label_ind += pad

        task_label_to_label_packet[task_name] = (preds, labels)
    return task_label_to_label_packet, task_label_to_boundaries


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
    error_analysis: bool,
    task_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray]],
    task_to_label_boundaries: Dict[str, Tuple[int, int]],
    split_name: str,
    dataset_ind: int,
    dataset: ClinicalNlpDataset,
    output_mode: Dict[str, str],
):
    task_to_error_inds: Dict[str, Union[None, np.ndarray]] = defaultdict(lambda: None)
    if error_analysis:
        for task, label_packet in task_to_label_packet.items():
            preds, labels = label_packet
            task_to_error_inds[task] = compute_disagreements(
                preds, labels, output_mode[task]
            )

    start_ind = end_ind = 0
    for ind in range(dataset_ind):
        start_ind += len(dataset.datasets[ind][split_name])
    end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    eval_dataset = Dataset.from_dict(
        dataset.processed_dataset[split_name][start_ind:end_ind]
    )

    if error_analysis:
        relevant_indices = set(
            chain.from_iterable(filter(None, task_to_error_inds.values()))
        )

    else:
        relevant_indices = set(range(len(eval_dataset["text"])))

    out_table = pd.DataFrame(
        columns=["text", *sorted(dataset.tasks)],
        index=relevant_indices,
    )

    out_table["text"] = [eval_dataset["text"][index] for index in relevant_indices]

    task2labels = dataset.get_labels()
    # for task_label, error_inds in task_to_error_inds.items():
    for task_name, packet in task_to_label_packet.items():
        preds, labels = packet
        torch_labels = np.array(eval_dataset["labels"])
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
    error_inds: Union[np.ndarray, None],
    torch_labels: np.ndarray,
) -> List[str]:
    if error_inds is not None and error_analysis:
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
        # since error_inds is now possibly None
        return len(all_torch_labels) * ["UNSUPPORTED TASK TYPE"]


def get_classification_prints(
    classification_labels: List[str],
    ground_truths: np.ndarray,  # List[str],
    task_predictions: np.ndarray,
) -> List[str]:
    resolved_predictions = task_predictions  # np.argmax(task_predictions, axis=1)
    predicted_labels = [*map(itemgetter(classification_labels), resolved_predictions)]

    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        return f"Ground: {ground} , Predicted {predicted}"

    return [*map(clean_string, zip(ground_truths, predicted_labels))]


def get_tagging_prints(
    tagging_labels: List[str],
    ground_truths: np.ndarray,  # List[str],
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    resolved_predictions = task_predictions  # np.argmax(task_predictions, axis=2)

    def human_readable_labels(index: int) -> str:
        # since resolved predictions is num errors x seq length
        # and torch labels is num errors x seq length x 1
        return " ".join(
            [
                tagging_labels[label_idx]
                for label_idx in resolved_predictions[index][
                    np.where(torch_labels[index].reshape(-1) != -100)
                ]
            ]
        )

    # do naive approach for now
    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        pred_str = " ".join(predicted)

        return f"Ground: {ground} , Predicted {pred_str}"

    return [
        *map(
            clean_string,
            zip(
                ground_truths,
                map(human_readable_labels, range(resolved_predictions.shape[0])),
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

import logging
import numpy as np
import pandas as pd
import re
import csv
import tqdm
import inspect

from datasets import Dataset
from transformers.trainer_utils import EvalPrediction
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from typing import Dict, List, Tuple, Union, Iterable
from itertools import chain
from operator import itemgetter
from collections import defaultdict

logger = logging.getLogger(__name__)


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
    output_mode: Dict[str, str],
):
    task_to_error_inds: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
    if error_analysis:
        for task, label_packet in tqdm.tqdm(
            task_to_label_packet.items(), desc=f"computing disagreements"
        ):
            preds, labels = label_packet
            task_to_error_inds[task] = compute_disagreements(
                preds, labels, output_mode[task]
            )

        relevant_indices: Iterable[int] = set(
            map(int, chain.from_iterable(task_to_error_inds.values()))
        )

    else:
        relevant_indices = range(len(eval_dataset["text"]))

    classification_tasks = filter(
        lambda t: output_mode[t] == classification, task_names
    )

    tagging_tasks = filter(lambda t: output_mode[t] == tagging, task_names)

    relex_tasks = filter(lambda t: output_mode[t] == relex, task_names)

    # ordering in terms of ease of reading
    out_table = pd.DataFrame(
        columns=["text", *classification_tasks, *tagging_tasks, *relex_tasks],
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
    for task_name, packet in tqdm.tqdm(
        task_to_label_packet.items(), desc="getting human readable labels"
    ):
        preds, labels = packet
        task_labels = task2labels[task_name]
        error_inds = task_to_error_inds[task_name]
        target_inds = error_inds if len(error_inds) > 0 else relevant_indices
        out_table[task_name][target_inds] = get_output_list(
            error_analysis,
            task_name,
            task_labels,
            task_to_label_boundaries,
            preds,
            labels,
            output_mode,
            error_inds,
            torch_labels,
            out_table["text"],
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
    text_column: pd.Series,
) -> List[str]:
    if len(error_inds) > 0 and error_analysis:
        ground_truth = labels[error_inds].astype(int)
        task_prediction = prediction[error_inds].astype(int)
        all_torch_labels = torch_labels[error_inds].astype(int)
        text_samples = pd.Series(text_column[error_inds])
    else:
        ground_truth = labels.astype(int)
        task_prediction = prediction.astype(int)
        all_torch_labels = torch_labels.astype(int)
        text_samples = text_column
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
            task_labels,
            ground_truth,
            task_prediction,
            task_torch_labels,
            text_samples,
            pred_task,
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
    predicted_labels = [classification_labels[index] for index in resolved_predictions]

    ground_strings = [classification_labels[index] for index in ground_truths]

    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        if ground == predicted:
            return "No error"
        return f"Ground: {ground} , Predicted {predicted}"

    return [*map(clean_string, zip(ground_strings, predicted_labels))]


def get_tagging_prints(
    tagging_labels: List[str],
    ground_truths: np.ndarray,
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
    text_samples: pd.Series,
    task_name: str,
) -> List[str]:
    resolved_predictions = task_predictions

    # since sometimes it's just
    # BIO with no suffixes and
    # we'll need to use the column name
    def get_ner_type(tag: str) -> str:
        elems = tag.split("-")
        if len(elems) > 1:
            return elems[-1].lower()
        return task_name.lower()

    def types2spans(
        raw_tag_inds: np.ndarray, token_ids: np.ndarray
    ) -> Dict[str, Tuple[int, int]]:

        type2inds = defaultdict(list)

        raw_labels = [
            tagging_labels[label_idx]
            for label_idx in raw_tag_inds[np.where(token_ids.reshape(-1) != -100)]
        ]

        for index, raw_label in enumerate(raw_labels):
            if raw_label != "O":
                type2inds[get_ner_type(raw_label)].append(index)

        return {
            ner_type: (inds[0], inds[-1] + 1) for ner_type, inds in type2inds.items()
        }

    def dictmerge(
        ground_dict: Dict[str, Tuple[int, int]], pred_dict: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Dict[str, Tuple[int, int]]]:
        disagreements = {}
        for key in {*ground_dict.keys(), *pred_dict.keys()}:
            if ground_dict[key] != pred_dict[key]:
                disagreements["ground"][key] = ground_dict[key]

                disagreements["predicted"][key] = pred_dict[key]

        return disagreements

    def get_out_string(
        disagreements: Dict[str, Dict[str, Tuple[int, int]]], instance: str
    ) -> str:
        instance_tokens = [*filter(None, instance.split())]

        ground_string = " , ".join(
            f'{key}: "{instance_tokens[span[0]:span[1]]}"'
            for key, span in disagreements["ground"].items()
        )

        predicted_string = " , ".join(
            f'{key}: "{instance_tokens[span[0]:span[1]]}"'
            for key, span in disagreements["predicted"].items()
        )

        if len(ground_string) == 0 == len(predicted_string):
            return "No errors"

        return f"Ground : {ground_string} Predicted : {predicted_string}"

    ground_span_dictionaries = (
        types2spans(ground_truth, torch_label)
        for ground_truth, torch_label in zip(ground_truths, torch_labels)
    )

    pred_span_dictionaries = (
        types2spans(pred, torch_label)
        for pred, torch_label in zip(resolved_predictions, torch_labels)
    )

    disagreement_dicts = (
        dictmerge(ground_dictionary, pred_dictionary)
        for ground_dictionary, pred_dictionary in zip(
            ground_span_dictionaries, pred_span_dictionaries
        )
    )

    # returning list instead of generator since pandas needs that
    return [
        get_out_string(disagreements, instance)
        for disagreements, instance in zip(disagreement_dicts, text_samples)
    ]


def get_relex_prints(
    relex_labels: List[str],
    ground_truths: np.ndarray,  # List[str],
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    Cell = Tuple[int, int, int]

    resolved_predictions = task_predictions  # np.argmax(task_predictions, axis=3)
    none_index = relex_labels.index("None") if "None" in relex_labels else -1

    def tuples_to_str(label_tuples: Iterable[Cell]):
        return [
            (row, col, relex_labels[label]) for row, col, label in sorted(label_tuples)
        ]

    def normalize_cells(
        raw_cells: np.ndarray, token_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # resolved_predictions[index]  shape is sent length x sent length
        # not the same shape insanity that we had for tagging

        (invalid_inds,) = np.where(np.diag(token_ids) != -100)
        if len(invalid_inds) > 0:
            # reset if this is an issue, but only want to report it
            # if there's an issue in the label, if it's an invalid
            # prediction that's just life
            token_ids[[range(len(token_ids)), range(len(token_ids))]] = -100
        # TODO find a more facile way to do this in numpy if possible
        reduced_prediction = np.array(
            [
                *filter(
                    len,
                    [
                        pred_row[np.where(token_row != -100)]
                        for pred_row, token_row in zip(raw_cells, token_ids)
                    ],
                )
            ]
        )
        return invalid_inds, reduced_prediction

    def find_disagreements(
        ground_pair: Tuple[np.ndarray, np.ndarray],
        pred_pair: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[Iterable[Cell], Iterable[Cell], Iterable[Cell]]:
        invalid_ground_inds, ground_matrix = ground_pair

        _, pred_matrix = pred_pair

        # should be commutative
        disagreements = np.where(ground_matrix != pred_matrix)

        bad_cells = zip(*invalid_ground_inds, ground_matrix[invalid_ground_inds])

        # nones will just clutter things up
        # and we will be able to infer disagreements on nones
        # from each other
        ground_cells = filter(
            lambda t: t[-1] == none_index,
            zip(*disagreements, ground_matrix[disagreements]),
        )

        pred_cells = filter(
            lambda t: t[-1] == none_index,
            zip(*disagreements, pred_matrix[disagreements]),
        )

        return bad_cells, ground_cells, pred_cells

    def to_string(
        bad_cells: Iterable[Cell],
        ground_cells: Iterable[Cell],
        pred_cells: Iterable[Cell],
    ) -> str:
        bad_cells_str = tuples_to_str(bad_cells)

        ground_cells_str = tuples_to_str(ground_cells)

        pred_cells_str = tuples_to_str(pred_cells)

        if len(ground_cells_str) == 0 == len(pred_cells_str):
            if len(bad_cells_str) > 0:
                return "INVALID RELATION LABELS : {bad_cells_str}"
            return "No errors"
        bad_out = (
            f"INVALID RELATION LABELS : {bad_cells_str} , "
            if len(bad_cells_str) > 0
            else ""
        )
        return f"{bad_out}Ground: {ground_cells_str} , Predicted : {pred_cells_str}"

    normalized_grounds = (
        normalize_cells(ground_truth, torch_label)
        for ground_truth, torch_label in zip(ground_truths, torch_labels)
    )

    normalized_preds = (
        normalize_cells(pred, torch_label)
        for pred, torch_label in zip(resolved_predictions, torch_labels)
    )

    disagreements = (
        find_disagreements(ground_pair, pred_pair)
        for ground_pair, pred_pair in zip(normalized_grounds, normalized_preds)
    )

    return [
        to_string(bad_cells, ground_cells, pred_cells)
        for bad_cells, ground_cells, pred_cells in disagreements
    ]

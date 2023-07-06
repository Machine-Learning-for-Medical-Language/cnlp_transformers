
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers.trainer_utils import EvalPrediction
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from typing import Dict, List, Tuple
from itertools import chain
from operator import itemgetter


def collect_disagreements(
    task_names: List[str],
    p: EvalPrediction,
    max_seq_length: int,
    output_modes,
    # ) -> Dict[int, Set[str]]:
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    # inds_to_labels = defaultdict(lambda: set())
    labels_to_inds = {}
    task_label_ind = 0

    tasklabel2labelboundaries = {}

    for task_ind, task_name in enumerate(task_names):
        tagger = output_modes[task_name] == tagging
        relations = output_modes[task_name] == relex
        if tagger:
            preds = np.argmax(p.predictions[task_ind], axis=2)
            # labels will be -100 where we don't need to tag
        elif relations:
            preds = np.argmax(p.predictions[task_ind], axis=3)
        else:
            preds = np.argmax(p.predictions[task_ind], axis=1)

        labels = np.array([])  # so the type checker doesn't complain

        if relations:
            # relation labels
            labels = p.label_ids[
                :, :, task_label_ind : task_label_ind + max_seq_length
            ].squeeze()
            tasklabel2labelboundaries[task_name] = (
                task_label_ind,
                task_label_ind + max_seq_length,
            )
            task_label_ind += max_seq_length
        elif p.label_ids.ndim == 3:
            if tagger:
                labels = p.label_ids[
                    :, :, task_label_ind : task_label_ind + 1
                ].squeeze()
            else:
                labels = p.label_ids[:, 0, task_label_ind].squeeze()
            # guessing second case is padded inds
            tasklabel2labelboundaries[task_name] = (task_label_ind, task_label_ind + 1)
            task_label_ind += 1
        elif p.label_ids.ndim == 2:
            labels = p.label_ids[:, task_ind].squeeze()
            # for prediction_index in compute_disagreements(
            #     preds,
            #     labels,
            #     dataset.output_modes[task_name],
            # ):
            #     inds_to_labels[prediction_index].a
        labels_to_inds[task_name] = compute_disagreements(
            preds,
            labels,
            output_modes[task_name],
        )
    return labels_to_inds, tasklabel2labelboundaries


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


def write_errors_for_dataset(
    output_fn,
    trainer,
    dataset,
    split_name,
    dataset_ind,
    output_mode,
    tokenizer,
):

    populate_errors_for_dataset(trainer, split_name, dataset_ind, dataset, output_mode)


def populate_errors_for_dataset(
    trainer,
    split_name: str,
    dataset_ind: int,
    dataset: ClinicalNlpDataset,
    output_mode: Dict[str, str],
    # inds2tasks: Dict[int, Set[str]],
    # labels2inds: Dict[str, np.ndarray],
):
    start_ind = end_ind = 0
    for ind in range(dataset_ind):
        start_ind += len(dataset.datasets[ind][split_name])
    end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    task2datasetind = {
        task_name: task_ind for task_ind, task_name in enumerate(dataset.tasks)
    }
    eval_dataset = Dataset.from_dict(
        dataset.processed_dataset[split_name][start_ind:end_ind]
    )
    raw_prediction = trainer.predict(test_dataset=eval_dataset)

    task2_error_inds, task2_label_boundaries = collect_disagreements(
        dataset.tasks, raw_prediction, dataset.args.max_seq_length, output_mode
    )
    # redundant but you never can tell
    relevant_indices = sorted(set(chain.from_iterable(task2_error_inds.values())))

    out_table = pd.DataFrame(
        columns=["text", *sorted(dataset.tasks)],
        index=relevant_indices,
    )

    out_table["text"] = [eval_dataset["text"][index] for index in relevant_indices]

    task2labels = dataset.get_labels()
    for task_label, error_inds in task2_error_inds.items():
        out_table[task_label][error_inds] = get_error_list(
            task_label,
            task2_label_boundaries,
            raw_prediction.predictions,
            task2labels,
            task2datasetind,
            output_mode,
            error_inds,
            eval_dataset,
        )
        print(f"Processed {len(error_inds)} for {task_label}")

        print(out_table[task_label])


# might be more efficient to return a pd.Series or something for the
# assignment and populate it via a generator but for now just use a list
def get_error_list(
    error_task: str,
    task2boundaries: Dict[str, Tuple[int, int]],
    prediction: EvalPrediction,
    task2labels: Dict[str, List[str]],
    task2ind: Dict[str, int],
    output_mode: Dict[str, str],
    error_inds: np.ndarray,
    eval_dataset: Dataset,
) -> List[str]:

    ground_truth = np.array(eval_dataset[error_task])[error_inds]
    task_prediction = prediction[task2ind[error_task]][error_inds]
    task_type = output_mode[error_task]
    task_labels = task2labels[error_task]
    # get the feeling this doesn't work for multiple tasks but we'll
    # probe those data structures when we run the code
    all_torch_labels = np.array(eval_dataset["label"])[
        error_inds
    ]  # should have only been indexing these?
    if task_type == classification:
        return get_classification_prints(task_labels, ground_truth, task_prediction)
    elif task_type == tagging:
        task_torch_labels = all_torch_labels
        if error_task in task2boundaries.keys():
            labels_start, labels_end = task2boundaries[error_task]
            task_torch_labels = all_torch_labels[:, :, labels_start:labels_end]
        return get_tagging_prints(
            task_labels, ground_truth, task_prediction, task_torch_labels
        )
    elif task_type == relex:
        task_torch_labels = all_torch_labels
        if error_task in task2boundaries.keys():
            labels_start, labels_end = task2boundaries[error_task]
            task_torch_labels = all_torch_labels[:, :, labels_start:labels_end]
        return get_relex_prints(
            task_labels, ground_truth, task_prediction, task_torch_labels
        )
    else:
        return len(error_inds) * ["UNSUPPORTED TASK TYPE"]


def get_classification_prints(
    classification_labels: List[str],
    ground_truths: List[str],
    task_predictions: np.ndarray,
) -> List[str]:
    resolved_predictions = np.argmax(task_predictions, axis=1)
    predicted_labels = [*map(itemgetter(classification_labels), resolved_predictions)]

    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        return f"Ground: {ground} , Predicted {predicted}"

    return [*map(clean_string, zip(ground_truths, predicted_labels))]


def get_tagging_prints(
    tagging_labels: List[str],
    ground_truths: List[str],
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    resolved_predictions = np.argmax(task_predictions, axis=2)

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
    ground_truths: List[str],
    task_predictions: np.ndarray,
    torch_labels: np.ndarray,
) -> List[str]:
    resolved_predictions = np.argmax(task_predictions, axis=3)
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

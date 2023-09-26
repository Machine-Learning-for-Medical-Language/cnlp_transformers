import logging
from typing import Any, Dict, Set

import numpy as np
from seqeval.metrics import classification_report as seq_cls
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from .cnlp_data import ClinicalNlpDataset
from .cnlp_processors import classification, mtl, relex, tagging

logger = logging.getLogger(__name__)


def fix_np_types(input_variable):
    """
    In the mtl classification setting, f1 is an array, and when the HF library
    tries to write out the training history to a json file it will throw an error.
    Here, we just check whether it's a numpy array and if so convert to a list.

    :meta private:
    """
    if isinstance(input_variable, np.ndarray):
        return list(input_variable)

    return input_variable


def tagging_metrics(
    label_set: Set[str],
    preds: np.ndarray,
    labels: np.ndarray,
    task_name: str,
) -> Dict[str, Any]:
    """
    One of the metrics functions for use in :func:`cnlp_compute_metrics`.

    Generates evaluation metrics for sequence tagging tasks.

    Ignores tags for which the true label is -100.

    The returned dict is structured as follows::

        {
            'acc': accuracy
            'token_f1': token-wise F1 score
            'f1': seqeval F1 score
            'report': seqeval classification report
        }

    :param label_set: The set of labels for this task
    :param preds: the predicted labels from the model
    :param labels: the true labels
    :param task_name: the name of the relevant task (unused)
    :return: a dictionary containing evaluation metrics
    """
    preds = preds.flatten()
    labels = labels.flatten().astype("int")

    pred_inds = np.where(labels != -100)
    preds = preds[pred_inds]
    labels = labels[pred_inds]

    pred_seq = [label_set[x] for x in preds]
    label_seq = [label_set[x] for x in labels]

    num_correct = (preds == labels).sum()

    acc = num_correct / len(preds)
    f1 = f1_score(labels, preds, average=None, zero_division=0)

    return {
        "acc": acc,
        "token_f1": fix_np_types(f1),
        "f1": fix_np_types(seq_f1([label_seq], [pred_seq])),
        "report": "\n" + seq_cls([label_seq], [pred_seq]),
    }


def relation_metrics(
    label_set: Set[str],
    preds: np.ndarray,
    labels: np.ndarray,
    task_name: str,
) -> Dict[str, Any]:
    """
    One of the metrics functions for use in :func:`cnlp_compute_metrics`.

    Generates evaluation metrics for relation extraction tasks.

    Ignores tags for which the true label is -100.

    The returned dict is structured as follows::

        {
            'f1': F1 score
            'acc': accuracy
            'recall': recall
            'precision': precision
        }

    :param label_set: the set of labels for this task
    :param preds: the predicted labels from the model
    :param labels: the true labels
    :return: a dictionary containing evaluation metrics
    """

    # If we are using the attention-based relation extractor, many impossible pairs
    # are set to -100 so pytorch loss functions ignore them. We need to make sure the
    # scorer also ignores them.
    relevant_inds = np.where(labels != -100)

    num_correct = (relevant_labels == relevant_preds).sum()
    acc = num_correct / len(relevant_preds)

    relevant_labels = [label_set[i] for i in labels[relevant_inds].astype("int")]
    relevant_preds = [label_set[i] for i in preds[relevant_inds].astype("int")]
    recall = recall_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    precision = precision_score(
        y_pred=relevant_preds, y_true=relevant_labels, average=None, zero_division=0
    )
    f1_scores = fix_np_types(
        f1_score(
            y_true=relevant_labels, y_pred=relevant_preds, average=None, zero_division=0
        )
    )
    report_dict = classification_report(
        y_true=relevant_labels, y_pred=relevant_preds, output_dict=True
    )
    report_str = classification_report(y_true=relevant_labels, y_pred=relevant_preds)

    return {
        "f1": f1_scores,
        "acc": acc,
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision),
        "report_dict": report_dict,
        "report_str": report_str,
    }


def acc_and_f1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    One of the metrics functions for use in :func:`cnlp_compute_metrics`.

    Generates evaluation metrics for generic tasks.

    The returned dict is structured as follows::

        {
            'acc': accuracy
            'f1': F1 score
            'acc_and_f1': mean of accuracy and F1 score
            'recall': recall
            'precision': precision
        }

    :param preds: the predicted labels from the model
    :param labels: the true labels
    :return: a dictionary containing evaluation metrics
    """
    acc = accuracy_score(y_pred=preds, y_true=labels)
    recall = recall_score(y_true=labels, y_pred=preds, average=None)
    precision = precision_score(
        y_true=labels, y_pred=preds, average=None, zero_division=0
    )
    f1 = f1_score(y_true=labels, y_pred=preds, average=None, zero_division=0)

    return {
        "acc": fix_np_types(acc),
        "f1": fix_np_types(f1),
        "acc_and_f1": fix_np_types((acc + f1) / 2),
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision),
    }


def cnlp_compute_metrics(
    task_name: str,
    preds: np.ndarray,
    labels: np.ndarray,
    output_mode: str,
    label_set: Set[str],
) -> Dict[str, Any]:
    """
    Function that defines and computes the metrics used for each task.

    When adding a task definition to this file, add a branch to this
    function defining what its evaluation metric invocation should be.
    If the new task is a simple classification task, a sensible default
    is defined; falling back on this will trigger a warning.

    :param task_name: the task name used to index into cnlp_processors
    :param preds: the predicted labels from the model
    :param labels: the true labels
    :param output_mode: the output mode of the classifier
    :param label_set: the set of output label names for the classifier
    :return: a dictionary containing evaluation metrics
    """

    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if output_mode == classification:
        return acc_and_f1(preds=preds, labels=labels)
    elif output_mode == tagging:
        return tagging_metrics(
            label_set, preds=preds, labels=labels, task_name=task_name
        )
    elif output_mode == relex:
        return relation_metrics(
            label_set, preds=preds, labels=labels, task_name=task_name
        )
    else:
        raise Exception(
            "There is no metric defined for this task in function cnlp_compute_metrics()"
        )

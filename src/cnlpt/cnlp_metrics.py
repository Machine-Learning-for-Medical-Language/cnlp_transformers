import logging
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, precision_score, classification_report, accuracy_score
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_cls
from .cnlp_processors import classification, mtl, tagging, relex


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

def tagging_metrics(processor, preds, labels, task_ind):
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

    :param DataProcessor processor: the data processor that was used to read in the data files
    :param numpy.ndarray preds: the predicted labels from the model
    :param numpy.ndarray labels: the true labels
    :rtype: typing.Dict[str, typing.Any]
    :return: a dictionary containing evaluation metrics
    """
    label_set = processor.get_labels()[task_ind]

    preds = preds.flatten()
    labels = labels.flatten().astype('int')

    pred_inds = np.where(labels != -100)
    preds = preds[pred_inds]
    labels = labels[pred_inds]

    pred_seq = [ label_set[x] for x in preds]
    label_seq = [ label_set[x] for x in labels]

    num_correct = (preds==labels).sum()

    acc = num_correct / len(preds)
    f1 = f1_score(labels, preds, average=None)

    return {'acc': acc, 'token_f1': fix_np_types(f1), 'f1': fix_np_types(seq_f1([label_seq], [pred_seq])), 'report':'\n'+seq_cls([label_seq], [pred_seq])}

def relation_metrics(processor, preds, labels, task_ind):
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

    :param DataProcessor processor: the data processor that was used to read in the data files
    :param numpy.ndarray preds: the predicted labels from the model
    :param numpy.ndarray labels: the true labels
    :rtype: typing.Dict[str, typing.Any]
    :return: a dictionary containing evaluation metrics
    """

    label_set = processor.get_labels()[task_ind]

    # If we are using the attention-based relation extractor, many impossible pairs
    # are set to -100 so pytorch loss functions ignore them. We need to make sure the
    # scorer also ignores them.
    relevant_inds = np.where(labels != -100)
    relevant_labels = labels[relevant_inds]
    relevant_preds = preds[relevant_inds]

    num_correct = (relevant_labels == relevant_preds).sum()
    acc = num_correct / len(relevant_preds)

    recall = recall_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    precision = precision_score(y_pred=relevant_preds, y_true=relevant_labels, average=None)
    f1_scores = fix_np_types(f1_score(y_true=relevant_labels, y_pred=relevant_preds, average=None))
    report_dict = classification_report(y_true=relevant_labels, y_pred=relevant_preds, output_dict=True)
    report_str = classification_report(y_true=relevant_labels, y_pred=relevant_preds)

    return {'f1': f1_scores, 'acc': acc, 'recall':fix_np_types(recall), 'precision':fix_np_types(precision), 'report_dict':report_dict, 'report_str':report_str }

def acc_and_f1(preds, labels):
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

    :param numpy.ndarray preds: the predicted labels from the model
    :param numpy.ndarray labels: the true labels
    :rtype: typing.Dict[str, typing.Any]
    :return: a dictionary containing evaluation metrics
    """
    acc = accuracy_score(y_pred=preds, y_true=labels)
    recall = recall_score(y_true=labels, y_pred=preds, average=None)
    precision = precision_score(y_true=labels, y_pred=preds, average=None)
    f1 = f1_score(y_true=labels, y_pred=preds, average=None)
    
    return {
        "acc": fix_np_types(acc),
        "f1": fix_np_types(f1),
        "acc_and_f1": fix_np_types((acc + f1) / 2),
        "recall": fix_np_types(recall),
        "precision": fix_np_types(precision)
    }

def cnlp_compute_metrics(task_name, task_ind, preds, labels, processor, output_mode):
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
        logger.warn("Choosing accuracy and f1 as default metrics; modify cnlp_compute_metrics() to customize for this task.")
        return acc_and_f1(preds=preds, labels=labels)
    elif output_mode == tagging:
        return tagging_metrics(processor, preds=preds, labels=labels, task_ind=task_ind)
    elif output_mode == relex:
        return relation_metrics(processor, preds=preds, labels=labels, task_ind=task_ind)
    else:
        raise Exception('There is no metric defined for this task in function cnlp_compute_metrics()')

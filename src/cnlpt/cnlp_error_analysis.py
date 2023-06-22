import numpy as np
from transformers.trainer_utils import EvalPrediction

from datasets import Dataset
from transformers import Trainer, PreTrainedTokenizer, EvalPrediction
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from typing import Dict, List


def collect_disagreements(
    task_names: List[str],
    p: EvalPrediction,
    max_seq_length: int,
    dataset: ClinicalNlpDataset,
):
    disagree_inds = {}
    task_label_ind = 0

    for task_ind, task_name in enumerate(task_names):
        tagger = dataset.output_modes[task_name] == tagging
        relations = dataset.output_modes[task_name] == relex
        if tagger:
            preds = np.argmax(p.predictions[task_ind], axis=2)
            # labels will be -100 where we don't need to tag
        elif relations:
            preds = np.argmax(p.predictions[task_ind], axis=3)
        else:
            preds = np.argmax(p.predictions[task_ind], axis=1)

        if relations:
            # relation labels
            labels = p.label_ids[
                :, :, task_label_ind : task_label_ind + data_args.max_seq_length
            ].squeeze()
            task_label_ind += data_args.max_seq_length
        elif p.label_ids.ndim == 3:
            if tagger:
                labels = p.label_ids[
                    :, :, task_label_ind : task_label_ind + 1
                ].squeeze()
            else:
                labels = p.label_ids[:, 0, task_label_ind].squeeze()
            task_label_ind += 1
        elif p.label_ids.ndim == 2:
            labels = p.label_ids[:, task_ind].squeeze()
        disagree_inds[task_name] = compute_disagreements(
            preds,
            labels,
            dataset.output_modes[task_name],
        )


def compute_disagreements(
    preds: np.ndarray,
    labels: np.ndarray,
    output_mode: str,
):
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
        return tagging_disagreements(
            preds=preds, labels=labels
        )
    elif output_mode == relex:
        return relation_disagreements(
            preds=preds, labels=labels,
        )
    else:
        raise Exception(
            "As yet unsupported task in cnlpt"
        )


def classification_disagreements(preds: np.ndarray, labels: np.ndarray):
    (indices,) = np.where(np.not_equal(preds, labels))
    return indices


def tagging_disagreements(preds: np.ndarray, labels: np.ndarray):
    (indices,) = np.where([*map(any, np.not_equal(preds, labels))])
    return indices


def relation_disagreements(preds: np.ndarray, labels: np.ndarray):
    (indices,) = np.where([*map(lambda s: s.any(), np.not_equal(preds, labels))])
    return indices


def write_errors_for_dataset(
    output_fn: str,
    trainer: Trainer,
    dataset: ClinicalNlpDataset,
    split_name: str,
    dataset_ind: int,
    output_mode: Dict[str, str],
):

    task_labels = dataset.get_labels()
    start_ind = end_ind = 0
    for ind in range(dataset_ind):
        start_ind += len(dataset.datasets[ind][split_name])
    end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    with open(output_fn, "w") as writer:
        eval_dataset = Dataset.from_dict(
            dataset.processed_dataset[split_name][start_ind:end_ind]
        )
        predictions = trainer.predict(test_dataset=eval_dataset).predictions
        for task_ind, task_name in enumerate(dataset.tasks):

            if output_mode[task_name] == classification:
                task_predictions = np.argmax(predictions[task_ind], axis=1)
                for index, item in enumerate(task_predictions):
                    item = task_labels[task_name][item]
                    writer.write(
                        "Task %d (%s) - Index %d - %s\n"
                        % (task_ind, task_name, index, item)
                    )
            elif output_mode[task_name] == tagging:
                task_predictions = np.argmax(predictions[task_ind], axis=2)
                tagging_labels = task_labels[task_name]
                for index, seq_pair in enumerate(zip(task_predictions, tagging_labels)):
                    pred_seq, true_seq = seq_pair
                    wpind_to_ind = {}
                    chunk_labels = []

                    token_inds = eval_dataset["input_ids"][index]
                    text = eval_dataset["text"][index]
                    predicted_labels = [
                        tagging_labels[task_predictions[index][i[0]]]
                        for i in filter(
                            lambda s: not all(i == -100 for i in s[1]),
                            enumerate(eval_dataset["label"][index]),
                        )
                    ]
                    true_ner = eval_dataset[task_name][index]

                    writer.write(
                        f"{eval_dataset.column_names} {text} : {len(text.split())} true ner {true_ner}  {predicted_labels} {len(predicted_labels)} \n"
                    )
            elif output_mode[task_name] == relex:
                task_predictions = np.argmax(predictions[task_ind], axis=3)
                relex_labels = task_labels[task_name]
                none_index = (
                    relex_labels.index("None") if "None" in relex_labels else -1
                )
                # assert task_labels[0] == 'None', 'The first labeled relation category should always be "None" but for task %s it is %s' % (task_names[task_ind], task_labels[0])

                for inst_ind in range(task_predictions.shape[0]):
                    inst_preds = task_predictions[inst_ind]
                    a1s, a2s = np.where(inst_preds != none_index)
                    for arg_ind in range(len(a1s)):
                        a1_ind = a1s[arg_ind]
                        a2_ind = a2s[arg_ind]
                        cat = relex_labels[inst_preds[a1_ind][a2_ind]]
                        writer.write(
                            "Task %d (%s) - Index %d - %s(%d, %d)\n"
                            % (task_ind, task_name, inst_ind, cat, a1_ind, a2_ind)
                        )
            else:
                raise NotImplementedError(
                    "Writing predictions is not implemented for this output_mode!"
                )

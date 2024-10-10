import logging
import re
from collections import defaultdict
from enum import Enum
from itertools import chain, groupby
from operator import itemgetter
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
from transformers import EvalPrediction

from .cnlp_processors import classification, relex, tagging

logger = logging.getLogger(__name__)


class SpanBegin(Enum):
    VALID = 0
    INVALID = 1

    def __str__(self) -> str:
        if self.value == 0:
            return ""
        return "WARNING: Invalid span beginning ( first token is I- )"


Cell = Tuple[int, int, int]
Span = Tuple[int, int, SpanBegin]


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def simple_softmax(x: list):
    """Softmax values for 1-D score array"""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def restructure_prediction(
    task_names: List[str],
    raw_prediction: EvalPrediction,
    max_seq_length: int,
    tagger: Dict[str, bool],
    relations: Dict[str, bool],
    output_prob: bool,
) -> Tuple[
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Dict[str, Tuple[int, int]],
]:
    task_label_ind = 0

    # disagreement collection stuff for this scope
    task_label_to_boundaries: Dict[str, Tuple[int, int]] = {}
    task_label_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = (
        {}
    )

    for task_ind, task_name in enumerate(task_names):
        preds, labels, pad, prob_values = structure_labels(
            raw_prediction,
            task_name,
            task_ind,
            task_label_ind,
            max_seq_length,
            tagger,
            relations,
            task_label_to_boundaries,
            output_prob,
        )
        task_label_ind += pad

        task_label_to_label_packet[task_name] = (preds, labels, prob_values)
    return (
        task_label_to_label_packet,
        task_label_to_boundaries,
    )


def structure_labels(
    p: EvalPrediction,
    task_name: str,
    task_ind: int,
    task_label_ind: int,
    max_seq_length: int,
    tagger: Dict[str, bool],
    relations: Dict[str, bool],
    task_label_to_boundaries: Dict[str, Tuple[int, int]],
    output_prob: bool,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    # disagreement collection stuff for this scope

    pad = 0
    prob_values: npt.NDArray[np.float64] = np.ndarray([])
    labels: npt.NDArray[np.int64] = np.ndarray([])
    if tagger[task_name]:
        preds = np.argmax(p.predictions[task_ind], axis=2)
        # labels will be -100 where we don't need to tag
    elif relations[task_name]:
        preds = np.argmax(p.predictions[task_ind], axis=3)
    else:
        preds = np.argmax(p.predictions[task_ind], axis=1)
        if output_prob:
            prob_values = np.max(
                [simple_softmax(logits) for logits in p.predictions[task_ind]], axis=1
            )

    # for inference
    if not hasattr(p, "label_ids") or p.label_ids is None:
        return preds, np.array([]), pad, np.array([])
    if relations[task_name]:
        # relation labels
        labels = p.label_ids[
            :, :, task_label_ind : task_label_ind + max_seq_length
        ].squeeze()
        task_label_to_boundaries[task_name] = (
            task_label_ind,
            task_label_ind + max_seq_length,
        )
        pad = max_seq_length
    elif p.label_ids.ndim == 3:
        if tagger[task_name]:
            labels = p.label_ids[:, :, task_label_ind : task_label_ind + 1].squeeze()
        else:
            labels = p.label_ids[:, 0, task_label_ind].squeeze()
        task_label_to_boundaries[task_name] = (task_label_ind, task_label_ind + 1)
        pad = 1
    elif p.label_ids.ndim == 2:
        labels = p.label_ids[:, task_ind].squeeze()

    return preds, labels, pad, prob_values


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
    elif output_mode == tagging or output_mode == relex:
        return relation_or_tagging_disagreements(preds=preds, labels=labels)
    else:
        raise Exception("As yet unsupported task in cnlpt")


def classification_disagreements(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    (indices,) = np.where(np.not_equal(preds, labels))
    return indices


def relation_or_tagging_disagreements(
    preds: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    (indices,) = np.where(
        [
            np.not_equal(pred[label != -100], label[label != -100]).any()
            for pred, label in zip(preds.astype(int), labels.astype(int))
        ]
    )
    return indices


def process_prediction(
    task_names: List[str],
    error_analysis: bool,
    output_prob: bool,
    character_level: bool,
    task_to_label_packet: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    eval_dataset,
    task_to_label_space: Dict[str, List[str]],
    output_mode: Dict[str, str],
) -> pd.DataFrame:
    task_to_error_inds: Dict[str, np.ndarray] = defaultdict(lambda: np.array([]))
    if error_analysis:
        for task, label_packet in tqdm.tqdm(
            task_to_label_packet.items(), desc="computing disagreements"
        ):
            preds, labels, prob_values = label_packet
            task_to_error_inds[task] = compute_disagreements(
                preds, labels, output_mode[task]
            )
        unique_indices = {
            int(i) for i in chain.from_iterable(task_to_error_inds.values())
        }
        # Indices need to be ordered otherwise Pandas
        # assumes the order in memory
        relevant_indices: Sequence[int] = sorted(unique_indices)

    else:
        unique_indices = set()
        relevant_indices = range(len(eval_dataset["text"]))

    classification_tasks = (
        task_name
        for task_name in task_names
        if output_mode[task_name] == classification
    )

    tagging_tasks = sorted(
        task_name for task_name in task_names if output_mode[task_name] == tagging
    )

    relex_tasks = sorted(
        task_name for task_name in task_names if output_mode[task_name] == relex
    )

    # ordering in terms of ease of reading
    out_table = pd.DataFrame(
        columns=[
            "text",
            *classification_tasks,
            *tagging_tasks,
            *relex_tasks,
        ],
        index=relevant_indices,
    )

    if len(relevant_indices) < len(eval_dataset["text"]):
        out_table["text"] = [eval_dataset["text"][index] for index in relevant_indices]
    else:
        out_table["text"] = list(eval_dataset["text"])
    out_table["text"] = out_table["text"].apply(remove_newline)

    out_table["text"] = out_table["text"].str.replace('"', "")
    out_table["text"] = out_table["text"].str.replace("//", "")
    out_table["text"] = out_table["text"].str.replace("\\", "")
    word_ids = eval_dataset["word_ids"]
    for task_name, packet in tqdm.tqdm(
        task_to_label_packet.items(), desc="getting human readable labels"
    ):
        preds, labels, prob_values = packet
        if not output_prob:
            prob_values = np.array([])
        task_labels = task_to_label_space[task_name]
        error_inds = task_to_error_inds[task_name]
        result_series = get_outputs(
            error_analysis=error_analysis,
            character_level=character_level,
            prob_values=prob_values,
            pred_task=task_name,
            task_labels=task_labels,
            prediction=preds,
            labels=labels,
            output_mode=output_mode,
            error_inds=error_inds,
            word_ids=word_ids,
            text_column=out_table["text"],
        )
        if len(error_inds) > 0:
            out_table[task_name][error_inds] = result_series
            remaining_indices = sorted(unique_indices - set(error_inds))
            out_table[task_name][remaining_indices] = len(remaining_indices) * [
                f"_no_{task_name}_errors_"
            ]
        else:
            out_table[task_name] = result_series
    return out_table


# might be more efficient to return a pd.Series or something for the
# assignment and populate it via a generator but for now just use a list
def get_outputs(
    error_analysis: bool,
    character_level: bool,
    prob_values: np.ndarray,
    pred_task: str,
    task_labels: List[str],
    prediction: np.ndarray,
    labels: np.ndarray,
    output_mode: Dict[str, str],
    error_inds: np.ndarray,
    word_ids: List[List[Union[None, int]]],
    text_column: pd.Series,
) -> pd.Series:
    if len(labels) > 0:
        if len(error_inds) > 0 and error_analysis:
            relevant_prob_values = (
                prob_values[error_inds]
                if output_mode[pred_task] == classification and len(prob_values) > 0
                else prob_values
            )
            ground_truth = labels[error_inds].astype(int)
            task_prediction = prediction[error_inds].astype(int)
            text_samples = pd.Series(text_column[error_inds])
            word_ids = [word_ids[error_ind] for error_ind in error_inds]
        else:
            relevant_prob_values = prob_values
            ground_truth = labels.astype(int)
            task_prediction = prediction.astype(int)
    else:
        relevant_prob_values = np.ndarray([])
        ground_truth = None
        task_prediction = prediction.astype(int)
    text_samples = text_column
    task_type = output_mode[pred_task]
    if task_type == classification:
        return get_classification_prints(
            pred_task, task_labels, ground_truth, task_prediction, relevant_prob_values
        )

    elif task_type == tagging:
        return get_tagging_prints(
            character_level,
            pred_task,
            task_labels,
            ground_truth,
            task_prediction,
            text_samples,
            word_ids,
        )
    elif task_type == relex:
        return get_relex_prints(
            pred_task, task_labels, ground_truth, task_prediction, word_ids
        )
    else:
        return pd.Series(len(error_inds) * ["UNSUPPORTED TASK TYPE"])


def get_classification_prints(
    task_name: str,
    classification_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    prob_values: np.ndarray,
) -> pd.Series:
    predicted_labels = (classification_labels[index] for index in task_predictions)

    def clean_string(gp: Tuple[str, str]) -> str:
        ground, predicted = gp
        if ground == predicted:
            return f"_{task_name}_error_detection_bug_"
        return f"Ground: {ground} Predicted: {predicted}"

    pred_list = predicted_labels
    if ground_truths is not None:
        ground_strings = [classification_labels[index] for index in ground_truths]

        pred_list = (clean_string(gp) for gp in zip(ground_strings, predicted_labels))

    if len(prob_values) > 0:
        return pd.Series(
            f"{pred} , Probability {prob:.6f}"
            for pred, prob in zip(pred_list, prob_values)
        )
    return pd.Series(pred_list)


def get_tagging_prints(
    character_level: bool,
    task_name: str,
    tagging_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    text_samples: pd.Series,
    word_ids: List[List[Union[None, int]]],
) -> pd.Series:
    # to save ourselves the branch instructions
    # in all the nested functions
    def get_tokens(inst: str) -> List[str]:
        return []

    token_sep = ""  # default since typesystem doesn't like the None
    if character_level:

        def get_tokens(inst: str) -> List[str]:
            return [token for token in inst if token is not None]

    else:

        def get_tokens(inst: str) -> List[str]:
            return [char for char in inst.split() if char is not None]

        token_sep = " "

    def flatten_dict(d: Dict[str, List[Span]]) -> Iterable[Tuple[str, Span]]:
        def tups(k: str, ls: Iterable[Span]) -> Iterable[Tuple[str, Span]]:
            return ((k, elem) for elem in ls)

        return chain.from_iterable(
            (((k, span) for k, span in tups(key, spans)) for key, spans in d.items())
        )

    def dict_to_str(d: Dict[str, List[Span]], tokens: List[str]) -> str:
        result = " , ".join(
            f'{key}: "{span[2]} {token_sep.join(tokens[span[0]:span[1]])}"'
            for key, span in flatten_dict(d)
        )
        return result

    # since sometimes it's just
    # BIO with no suffixes and
    # we'll need to use the column name
    def get_ner_type(tag: str) -> str:
        elems = tag.split("-")
        if len(elems) > 1:
            return elems[-1].lower()
        return task_name.lower()

    # NER model output tags without NER task info (e.g. B-fxno -> B)
    def get_partitions(annotation: List[str]) -> str:
        return "".join(tag[0].upper() for tag in annotation)

    # Group B's individually, B's followed by any number of I's,
    # or any number of I's by themselves with no B's e.g.
    # OOOOOOBBBBBBBIIIIBIBIBI
    # -> OOOOOO B B B B B B BIIII BI BI BI
    # OOOOIIO
    # -> OOOO II O
    # The latter is a pathological case that
    # we run into only occasionally
    def process_labels(annotation: List[str]) -> Iterable[Span]:
        span_begin, span_end = 0, 0
        partitions = get_partitions(annotation)
        for tag_group in filter(None, re.split(r"(B?I*)|(O+)", partitions)):
            span_end = len(tag_group) + span_begin - 1
            valid_begin = tag_group[0] == "B"
            invalid_begin = tag_group[0] == "I"
            if valid_begin or invalid_begin:
                # Get indices in list/string of each span
                # which describes a mention
                span = (
                    span_begin,
                    span_end + 1,
                    SpanBegin.VALID if valid_begin else SpanBegin.INVALID,
                )
                yield span
            span_begin = span_end + 1

    def types_to_spans(
        raw_tag_inds: np.ndarray, word_ids: List[Union[None, int]]
    ) -> Dict[str, List[Span]]:
        # def types_to_spans(idx, src, tup):
        # raw_tag_inds, word_ids = tup
        relevant_token_ids_and_tags = [
            (word_id, tag)
            for word_id, tag in zip(word_ids, raw_tag_inds)
            if word_id is not None
        ]
        relevant_token_ids_and_tags = [
            next(group)
            for _, group in groupby(relevant_token_ids_and_tags, key=itemgetter(0))
        ]

        raw_labels = [tagging_labels[tag] for _, tag in relevant_token_ids_and_tags]
        span_tuples = [
            (get_ner_type(raw_labels[tup[0]]), tup)
            for tup in process_labels(raw_labels)
        ]
        type_to_spans = {
            ner_type: [g[1] for g in group]
            for ner_type, group in groupby(
                sorted(span_tuples, key=itemgetter(0)), key=itemgetter(0)
            )
        }
        return type_to_spans

    def dictmerge(
        ground_dict: Dict[str, List[Span]],
        pred_dict: Dict[str, List[Span]],
    ) -> Dict[str, Dict[str, List[Span]]]:
        disagreements: Dict[str, Dict[str, List[Span]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key in {*ground_dict.keys(), *pred_dict.keys()}:
            ground_spans = ground_dict.get(key, [])
            pred_spans = pred_dict.get(key, [])

            disagreements["ground"][key].extend(
                span for span in ground_spans if span not in pred_spans
            )

            disagreements["predicted"][key].extend(
                span for span in pred_spans if span not in ground_spans
            )

        def empty_disagree(key: str) -> bool:
            return (
                len(disagreements["predicted"][key])
                == 0
                == len(disagreements["ground"][key])
            )

        if all(empty_disagree(key) for key in {*ground_dict.keys(), *pred_dict.keys()}):
            logger.warning(
                f"Empty disagreements, here are the ground vs pred spans {ground_dict} {pred_dict}"
            )

        return disagreements

    def get_error_out_string(
        disagreements: Dict[str, Dict[str, List[Span]]],
        instance: str,
    ) -> str:
        instance_tokens = get_tokens(instance)
        ground_string = dict_to_str(disagreements["ground"], instance_tokens)

        predicted_string = dict_to_str(disagreements["predicted"], instance_tokens)

        if len(ground_string) == 0 == len(predicted_string):
            return f"_{task_name}_error_detection_bug_"

        return f"Ground: {ground_string} Predicted: {predicted_string}"

    def get_pred_out_string(type_to_spans: Dict[str, List[Span]], instance: str) -> str:
        instance_tokens = get_tokens(instance)
        result = dict_to_str(type_to_spans, instance_tokens)
        return result

    pred_span_dictionaries = (
        types_to_spans(pred, word_id_ls)
        for pred, word_id_ls in zip(task_predictions, word_ids)
    )

    def debug(
        pred: np.ndarray, ground: np.ndarray, word_id_ls: List[Union[None, int]]
    ) -> None:
        relevant_indices = np.where(ground != 100)
        assert np.not_equal(
            pred[relevant_indices], ground[relevant_indices]
        ).any(), f"Defcon 1 - samples are equal at the ndarray level {pred} {ground}"
        relevant_pred_ids_and_tags = [
            (word_id, ptag)
            for word_id, ptag, gtag in zip(word_id_ls, pred, ground)
            if gtag != -100
            # for word_id, tag in zip(word_id_ls, pred)
            # if word_id is not None
        ]

        relevant_ground_ids_and_tags = [
            (word_id, tag)
            for word_id, tag in zip(word_id_ls, ground)
            if tag != -100
            # if word_id is not None
        ]
        assert (
            relevant_pred_ids_and_tags != relevant_ground_ids_and_tags
        ), f"Defcon 2 - samples are equal at the word id filtered tag level {relevant_pred_ids_and_tags} {relevant_ground_ids_and_tags}"
        grouped_ground_ids_and_tags = [
            next(group)
            for _, group in groupby(relevant_ground_ids_and_tags, key=itemgetter(0))
        ]
        grouped_pred_ids_and_tags = [
            next(group)
            for _, group in groupby(relevant_pred_ids_and_tags, key=itemgetter(0))
        ]

        assert grouped_pred_ids_and_tags != grouped_ground_ids_and_tags, (
            "Defcon 3 - samples are equal at the raw group level\n\n"
            f"grouped pred ids:{grouped_pred_ids_and_tags}\n\ngrouped ground ids:{grouped_ground_ids_and_tags}"
            f"\n\nrelevant pred ids:{relevant_pred_ids_and_tags}\n\nrelevant ground ids:{relevant_ground_ids_and_tags}\n\n"
            f"raw pred {pred}\n\nraw ground {ground}\n\nraw word id ls{word_id_ls}"
        )
        pred_raw_labels = [tagging_labels[tag] for _, tag in grouped_pred_ids_and_tags]
        pred_span_tuples = [
            (get_ner_type(pred_raw_labels[tup[0]]), tup)
            for tup in process_labels(pred_raw_labels)
        ]
        pred_types_to_spans = {
            ner_type: [g[1] for g in group]
            for ner_type, group in groupby(
                sorted(pred_span_tuples, key=itemgetter(0)), key=itemgetter(0)
            )
        }

        ground_raw_labels = [
            tagging_labels[tag] for _, tag in grouped_ground_ids_and_tags
        ]
        ground_span_tuples = [
            (get_ner_type(ground_raw_labels[tup[0]]), tup)
            for tup in process_labels(ground_raw_labels)
        ]
        ground_types_to_spans = {
            ner_type: [g[1] for g in group]
            for ner_type, group in groupby(
                sorted(ground_span_tuples, key=itemgetter(0)), key=itemgetter(0)
            )
        }
        assert (
            pred_raw_labels != ground_raw_labels
        ), f"Defcon 4 - samples are equal at the raw labels level {pred_raw_labels} {ground_raw_labels}"
        assert (
            pred_span_tuples != ground_span_tuples
        ), f"Defcon 5 - samples are equal at the span tuples level {pred_span_tuples} {ground_span_tuples}"
        assert (
            pred_types_to_spans != ground_types_to_spans
        ), f"Defcon 6 - samples are equal at the type to span level {pred_types_to_spans} {ground_types_to_spans}"

    if ground_truths is not None:
        for pred, ground, word_id_ls in zip(task_predictions, ground_truths, word_ids):
            debug(pred, ground, word_id_ls)
        return pd.Series([])
        # ground_span_dictionaries = (
        #     types_to_spans(ground_truth, word_id_ls)
        #     for ground_truth, word_id_ls in zip(ground_truths, word_ids)
        # )
        # disagreement_dicts = (
        #     dictmerge(ground_dictionary, pred_dictionary)
        #     for ground_dictionary, pred_dictionary in zip(
        #         ground_span_dictionaries, pred_span_dictionaries
        #     )
        # )

        # return pd.Series(
        #     get_error_out_string(disagreements, instance)
        #     for disagreements, instance in zip(disagreement_dicts, text_samples)
        # )

    return pd.Series(
        get_pred_out_string(type_to_pred_spans, instance)
        for type_to_pred_spans, instance in zip(pred_span_dictionaries, text_samples)
    )


def get_relex_prints(
    task_name: str,
    relex_labels: List[str],
    ground_truths: Union[None, np.ndarray],
    task_predictions: np.ndarray,
    word_ids: List[List[Union[None, int]]],
) -> pd.Series:
    resolved_predictions = task_predictions
    none_index = relex_labels.index("None") if "None" in relex_labels else -1

    # thought we'd filtered them out but apparently not
    def tuples_to_str(label_tuples: Iterable[Cell]) -> str:
        return " ".join(
            f"( {row}, {col}, {relex_labels[label]} )"
            for row, col, label in sorted(label_tuples)
        )

    def normalize_cells(
        raw_cells: np.ndarray, token_ids: List[Union[None, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        (invalid_inds,) = np.where(np.diag(raw_cells) != -100)
        # just in case

        word_ids_and_indices = [
            (index, word_id)
            for index, word_id in enumerate(token_ids)
            if word_id is not None
        ]

        wordpeice_collapsed = (
            next(group)
            for _, group in groupby(word_ids_and_indices, key=lambda s: s[1])
        )

        relevant_indices_iter, _ = zip(*wordpeice_collapsed)

        relevant_indices_ls = list(relevant_indices_iter)

        reduced_matrix = np.array(
            [raw_cells[index][relevant_indices_ls] for index in relevant_indices_ls]
        )

        np.fill_diagonal(reduced_matrix, none_index)

        assert (
            reduced_matrix.shape[0] == reduced_matrix.shape[1]
        ), f"reduced matrix shape: {reduced_matrix.shape}"

        assert -100 not in reduced_matrix, f"final matrix contents: {reduced_matrix}"

        return invalid_inds, reduced_matrix

    def find_disagreements(
        ground_pair: Tuple[np.ndarray, np.ndarray],
        pred_pair: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[Iterable[Cell], Iterable[Cell], Iterable[Cell]]:
        invalid_ground_inds, ground_matrix = ground_pair

        _, pred_matrix = pred_pair

        disagreements = np.where(ground_matrix != pred_matrix)

        if len(ground_matrix) == 0 == len(pred_matrix) == len(invalid_ground_inds):
            return [], [], []

        bad_cells = (
            [
                (*i, j)
                for i, j in zip(
                    zip(invalid_ground_inds, invalid_ground_inds),
                    ground_matrix[invalid_ground_inds, invalid_ground_inds],
                )
            ]
            if len(invalid_ground_inds) > 0
            else []
        )
        # nones will just clutter things up
        # and we will be able to infer disagreements on nones
        # from each other
        ground_cells = [
            cell
            for cell in zip(*disagreements, ground_matrix[disagreements])
            if cell[-1] != none_index
        ]

        pred_cells = [
            cell
            for cell in zip(*disagreements, pred_matrix[disagreements])
            if cell[-1] != none_index
        ]

        return bad_cells, ground_cells, pred_cells

    def to_error_string(
        bad_cells: Iterable[Cell],
        ground_cells: Iterable[Cell],
        pred_cells: Iterable[Cell],
    ) -> str:
        bad_cells_str = tuples_to_str(bad_cells)

        ground_cells_str = tuples_to_str(ground_cells)

        pred_cells_str = tuples_to_str(pred_cells)
        if len(ground_cells_str) == 0 == len(pred_cells_str):
            bad_cells_msg = (
                "INVALID RELATION LABELS : {bad_cells_str} "
                if len(bad_cells_str) > 0
                else ""
            )
            return f"{bad_cells_msg}{task_name}_error_detection_bug_"
        bad_out = (
            f"INVALID RELATION LABELS : {bad_cells_str} , "
            if len(bad_cells_str) > 0
            else ""
        )
        return f"{bad_out}Ground: {ground_cells_str} Predicted: {pred_cells_str}"

    def to_pred_string(reduced_matrix: np.ndarray) -> str:
        non_none_inds = np.where(reduced_matrix != none_index)
        non_none_cell_tuples = zip(
            *non_none_inds, reduced_matrix[non_none_inds].astype(int)
        )
        return tuples_to_str(non_none_cell_tuples)

    normalized_pred_pairs = (
        normalize_cells(pred, word_id_ls)
        for pred, word_id_ls in zip(resolved_predictions, word_ids)
    )
    if ground_truths is not None:
        normalized_ground_pairs = (
            normalize_cells(ground_truth, word_id_ls)
            for ground_truth, word_id_ls in zip(ground_truths, word_ids)
        )
        disagreements = (
            find_disagreements(ground_pair, pred_pair)
            for ground_pair, pred_pair in zip(
                normalized_ground_pairs, normalized_pred_pairs
            )
        )

        return pd.Series(
            to_error_string(bad_cells, ground_cells, pred_cells)
            for bad_cells, ground_cells, pred_cells in disagreements
        )
    return pd.Series(
        to_pred_string(reduced_pred_matrix)
        for _, reduced_pred_matrix in normalized_pred_pairs
    )

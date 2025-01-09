import logging
from collections import deque
from collections.abc import Iterable
from typing import Union

import numpy as np
from transformers import BatchEncoding
from transformers.tokenization_utils import PreTrainedTokenizer

from .features import HierarchicalInputFeatures, InputFeatures
from .tasks import classification, relex, tagging

logger = logging.getLogger(__name__)
none_column = "__None__"


def cnlp_convert_features_to_hierarchical(
    features: BatchEncoding,
    chunk_len: int,
    num_chunks: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    insert_empty_chunk_at_beginning: bool = False,
) -> BatchEncoding:
    """
    Chunk an instance of InputFeatures into an instance of HierarchicalInputFeatures
    for the hierarchical model.

    :param features: the dictionary containing mappings from properties to lists of values for each instance for each of those properties
    :param chunk_len: the maximum length of a chunk
    :param num_chunks: the maximum number of chunks in the instance
    :param cls_id: the tokenizer's ID representing the CLS token
    :param sep_id: the tokenizer's ID representing the SEP token
    :param pad_id: the tokenizer's ID representing the PAD token
    :param insert_empty_chunk_at_beginning: whether to insert an
        empty chunk at the beginning of the instance
    :return: an instance of :class:`transformers.BatchEncoding` containing the chunked instance
    """

    for ind in range(len(features["input_ids"])):
        # Get feature variables
        # input_ids_, attention_mask_, token_type_ids_, event_tokens_, label_ = astuple(features)
        input_ids_ = features["input_ids"][ind]
        attention_mask_ = features["attention_mask"][ind]
        token_type_ids_ = features.get("token_type_ids", None)
        if token_type_ids_ is not None:
            token_type_ids_ = token_type_ids_[ind]
        event_tokens_ = features["event_mask"][ind]

        assert len(input_ids_) == len(attention_mask_) == len(event_tokens_)

        # Split the sample's tokens into several chunk lists.
        chunks = []
        if attention_mask_ is not None:
            chunks_attention_mask = []
        else:
            chunks_attention_mask = None
        if token_type_ids_ is not None:
            chunks_token_type_ids = []
        else:
            chunks_token_type_ids = None
        if event_tokens_ is not None:
            chunks_event_tokens = []
        else:
            chunks_event_tokens = None

        def pad_chunk(chunk, pad_type=pad_id):
            return chunk + [pad_type] * (chunk_len - len(chunk))

        def format_chunk(
            chunk, cls_type=cls_id, sep_type=sep_id, pad_type=pad_id, pad=True
        ):
            formatted_chunk = [cls_type] + chunk + [sep_type]
            if pad:
                return pad_chunk(formatted_chunk, pad_type=pad_type)
            else:
                return formatted_chunk

        start = 1

        while True:
            if start >= len(input_ids_) or input_ids_[start] in {pad_id, sep_id}:
                # we have moved past the end of the sequence or have reached
                #  either the SEP token or a PAD token.
                break

            # end right before where the SEP token will go
            end = min(start + chunk_len - 2, len(input_ids_))

            # if we are ending on a PAD token or the SEP token, end before the SEP token
            if (
                input_ids_[end - 1] in {pad_id, sep_id}
                and sep_id in input_ids_[start:end]
            ):
                end = input_ids_.index(sep_id, start, end)

            chunks.append(format_chunk(input_ids_[start:end]))
            if chunks_attention_mask is not None:
                chunks_attention_mask.append(
                    format_chunk(attention_mask_[start:end], cls_type=1, sep_type=1)
                )
            if chunks_token_type_ids is not None:
                chunks_token_type_ids.append(
                    format_chunk(token_type_ids_[start:end], cls_type=0, sep_type=0)
                )
            if chunks_event_tokens is not None:
                chunks_event_tokens.append(
                    format_chunk(event_tokens_[start:end], cls_type=1, sep_type=1)
                )

            start = end

        def create_pad_chunk(cls_type=cls_id, sep_type=sep_id, pad_type=pad_id):
            return pad_chunk([cls_type] + [sep_type], pad_type=pad_type)

        # Insert an empty chunk at the beginning.
        if insert_empty_chunk_at_beginning:
            chunks.insert(0, create_pad_chunk())
            if chunks_attention_mask is not None:
                chunks_attention_mask.insert(0, create_pad_chunk(1, 1, 0))
            if chunks_token_type_ids is not None:
                # TODO: do we want special TTIDs?
                chunks_token_type_ids.insert(0, create_pad_chunk(0, 0, 0))
            if chunks_event_tokens is not None:
                # TODO: do we want special ETs?
                chunks_event_tokens.insert(0, create_pad_chunk(1, 1, 0))

        # Truncate the chunks and add attention masks
        chunks = chunks[:num_chunks]
        if chunks_attention_mask is not None:
            chunks_attention_mask = chunks_attention_mask[:num_chunks]
        if chunks_token_type_ids is not None:
            chunks_token_type_ids = chunks_token_type_ids[:num_chunks]
        if chunks_event_tokens is not None:
            chunks_event_tokens = chunks_event_tokens[:num_chunks]

        # Add empty lists to list of chunks, if the number of chunks less than max number.
        while len(chunks) < num_chunks:
            chunks.append(create_pad_chunk())
            if chunks_attention_mask is not None:
                chunks_attention_mask.append(create_pad_chunk(1, 1, 0))
            if chunks_token_type_ids is not None:
                chunks_token_type_ids.append(create_pad_chunk(0, 0, 0))
            if chunks_event_tokens is not None:
                chunks_event_tokens.append(create_pad_chunk(1, 1, 0))

        features["input_ids"][ind] = chunks
        features["attention_mask"][ind] = chunks_attention_mask
        if token_type_ids_ is not None:
            features["token_type_ids"][ind] = chunks_token_type_ids
        features["event_mask"][ind] = chunks_event_tokens

    return features


def cnlp_preprocess_data(
    examples: dict[str, Union[list[str], list[int], list[float]]],
    tokenizer: PreTrainedTokenizer,
    max_length: Union[int, None] = None,
    tasks: list[str] = [],
    label_lists: Union[dict[str, list[str]], None] = None,
    output_modes: Union[dict[str, str], None] = None,
    inference: bool = False,
    hierarchical: bool = False,
    chunk_len: int = -1,
    num_chunks: int = -1,
    character_level: bool = False,
    insert_empty_chunk_at_beginning: bool = False,
    truncate_examples: bool = False,
) -> Union[list[InputFeatures], list[HierarchicalInputFeatures]]:
    """
    Processes the dictionary of data inputs created by
    the processor defined in :data:`cnlpt.cnlp_processors.cnlp_processors`
    and converts the examples into a list of :class:`InputFeatures` or
    :class:`HierarchicalInputFeatures`, depending on the model.

    :param examples:
        the dictionary containing the input data to convert
    :param tokenizer: the tokenizer
    :param max_length: the maximum sequence length
        at which to truncate examples
    :param tasks: the task name(s) in a list, used to index the labels in the examples list.
    :param label_lists: a mapping from
        tasks to the list of labels for each task. If not provided explicitly, it will be retrieved from
        the processor with :meth:`transformers.DataProcessor.get_labels`.
    :param output_modes: the output modes for this task.
        If not provided explicitly, it will be retrieved from
        :data:`cnlpt.cnlp_processors.cnlp_output_modes`.
    :param inference: whether we're doing training or inference only -- if inference mode the labels associated with examples can't be trusted.
    :param hierarchical: whether to structure the data for the hierarchical
        model (:class:`cnlpt.HierarchicalTransformer.HierarchicalModel`)
    :param chunk_len: for the hierarchical model, the length of each
        chunk in tokens
    :param num_chunks: for the hierarchical model, the number of chunks
    :param insert_empty_chunk_at_beginning: for the hierarchical model,
        whether to insert an empty chunk at the beginning of the list of chunks
        (equivalent in theory to a CLS chunk).
    :param truncate_examples: whether to truncate the string representation
        of the example instances printed to the log
    :return: the list of converted input features
    """
    character_level = type(tokenizer).__name__ == "CanineTokenizer"

    if max_length is None:
        max_length = tokenizer.max_len

    # Try to infer the structure based on column names
    if "text" in examples.keys():
        if character_level:
            sentences = list(examples["text"])
        else:
            sentences = [str(example).split(" ") for example in examples["text"]]
        num_instances = len(examples["text"])
    elif "text_b" in examples.keys():
        # FIXME - not sure if this is right but doesn't get used much in our data
        raise NotImplementedError(
            "2-sentence classification has not been re-implemented yet."
        )
        sentences = (examples["text_a"], examples["text_b"])
    else:
        raise Exception(
            'The data does not seem to have a text column (literally a column labeled "text" is required)'
        )

    if hierarchical:
        padding = False
    else:
        padding = "max_length"

    result = tokenizer(
        sentences,
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=not character_level,
    )

    special_token_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.sep_token_id,
        tokenizer.cls_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    # slow tokenizers -> build your own word ids

    else:
        if character_level:

            def get_word_ids(indices: Iterable[int]) -> list[Union[int, None]]:
                current = 0
                raw: deque[Union[int, None]] = deque()
                for index in indices:
                    if index in special_token_ids:
                        raw.append(None)
                    else:
                        raw.append(current)
                        current += 1
                return list(raw)

            result["word_ids"] = [
                get_word_ids(indices) for indices in result["input_ids"]
            ]
        else:
            ValueError(
                f"{type(tokenizer).__name__}"
                "is a slow ( non-Rust ) tokenizer and thus word_ids is not implemented by default, "
                "you can provide your own implementation for extracting word_ids "
                "( see  https://huggingface.co/docs/tokenizers/main/en/api/encoding#tokenizers.Encoding.word_ids) for "
                "your model in this file"
            )
    # Now that we have the labels for each instances, and we've tokenized the input sentences,
    # we need to solve the problem of aligning labels with word piece indexes for the tasks of tagging
    # (which has one label per pre-wordpiece token) and relations (which are defined as tuples which
    # contain pre-wordpiece token indices)
    if not inference:
        assert (
            label_lists is not None and output_modes is not None
        ), f"label_lists {label_lists} output_modes {output_modes} must both be non-None"
        # Create a label map for each task in this dataset: { task1 => {label_0: 0, label_1: 1, label_2:, 2}, task2 => {label_0: 0, label_1:1} }
        label_map = {
            task: {label: i for i, label in enumerate(label_lists[task])}
            for task in label_lists.keys()
        }
        for task in tasks:
            if none_column in label_map[task]:
                raise Exception(
                    f"There is a column named {none_column} which is a reserved name"
                )
            label_map[task][none_column] = -100

        raw_labels = []
        labels = []

        # Create a list of mapped labels for every task in this dataset, with different mapping tactics for different types,
        # classification vs tagging vs. relations.
        for task_ind, task in enumerate(tasks):
            task_labels = []
            raw_labels.append(examples[task])

            if output_modes[task] == classification:
                task_labels = [label_map[task][label] for label in raw_labels[task_ind]]
                # labels is just a list of one label for each instance
            elif output_modes[task] == tagging:
                task_labels = [
                    [label_map[task][label] for label in inst_labels.split()]
                    for inst_labels in raw_labels[task_ind]
                ]
                # labels is a list of lists, where each internal list is the set of tags for that instance.
            elif output_modes[task] == relex:
                for inst_rels in raw_labels[task_ind]:
                    if inst_rels is None or inst_rels == "None":
                        task_labels.append(["None"])
                    else:
                        # The label for a sentence with multiple relations looks like this:
                        # (105,109,OVERLAP) , (64,66,CONTAINS) , (100,105,CONTAINS) , (81,88,CONTAINS) , (81,95,CONTAINS) , (105,106,OVERLAP) , (81,100,CONTAINS)
                        # Split into relations, then remove parens and split with commas into relation components (start offset, end offset, category)
                        inst_labels = []
                        for rel in inst_rels.split(" , "):
                            start_token, end_token, category = rel[1:-1].split(",")
                            inst_labels.append(
                                (
                                    int(start_token),
                                    int(end_token),
                                    label_map[task].get(category, 0),
                                )
                            )
                        task_labels.append(inst_labels)
            else:
                raise NotImplementedError(
                    f"This method is not complete for output mode {output_modes[task]}"
                )
            labels.append(task_labels)

        # Convert the labels to column format that arrow prefers
        labels = list(zip(*labels))

        result["label"] = _build_pytorch_labels(
            result,
            tasks,
            labels,
            output_modes,
            num_instances,
            max_length,
            label_lists,
        )
    if not character_level:
        result["event_mask"] = _build_event_mask_word_piece(
            result,
            num_instances,
            tokenizer.convert_tokens_to_ids("<e>"),
            tokenizer.convert_tokens_to_ids("</e>"),
        )
    else:
        result["event_mask"] = _build_event_mask_character(
            result,
            num_instances,
        )
    if hierarchical:
        result = cnlp_convert_features_to_hierarchical(
            result,
            chunk_len=chunk_len,
            num_chunks=num_chunks,
            cls_id=tokenizer.cls_token_id,
            sep_id=tokenizer.sep_token_id,
            pad_id=tokenizer.pad_token_id,
            insert_empty_chunk_at_beginning=insert_empty_chunk_at_beginning,
        )

    return result


def _build_event_mask_word_piece(
    result: BatchEncoding, num_insts: int, event_start_token_id, event_end_token_id
):
    """
    Create arrays corresponding to input tokens where the events to be classified contain special mask tokens.
    These are used if the --event flag is specified to classify event tokens rather than the [CLS] token.
    :param result: The input encodings of the tokens
    :param num_insts: The length of the input
    :param event_start_token_id: The special token index used to indicate the start of an event.
    :param event_end_token_id: The special token index used to indicate the end of an event.
    :return: The list of lists of per-instance event mask values corresponding to the input tokens.
    """
    event_tokens = []
    for i in range(num_insts):
        input_ids = result["input_ids"][i]
        try:
            event_start = input_ids.index(event_start_token_id)
        except Exception:
            event_start = -1

        try:
            event_end = input_ids.index(event_end_token_id)
        except Exception:
            event_end = len(input_ids) - 1

        if event_start >= 0:
            inst_event_tokens = (
                [0] * event_start
                + [1] * (event_end - event_start + 1)
                + [0] * (len(input_ids) - event_end - 1)
            )
        else:
            inst_event_tokens = [1] * len(input_ids)

        event_tokens.append(inst_event_tokens)

    return event_tokens


def _build_event_mask_character(result: BatchEncoding, num_insts: int):
    event_tokens = []
    for i in range(num_insts):
        input_ids = result["input_ids"][i]
        inst_event_tokens = [1] * len(input_ids)
        event_tokens.append(inst_event_tokens)

    return event_tokens


def _build_pytorch_labels(
    result: BatchEncoding,
    tasks: list[str],
    labels: list,
    output_modes: dict[str, str],
    num_instances: int,
    max_length: int,
    label_lists: dict[str, list[str]],
):
    # labels_out = []
    # TODO -- also adapt to character level
    pad_classification = False
    if relex in output_modes.values() or tagging in output_modes.values():
        # we have tagging as the highest dimensional output
        max_dims = 2
        if classification in output_modes.values():
            pad_classification = True
    else:
        # classification only
        max_dims = 1

    def build_labels_for_task(task_ind, task):
        return _build_labels_for_task(
            task,
            task_ind,
            output_modes,
            result,
            labels,
            num_instances,
            max_length,
            label_lists,
            pad_classification,
        )

    labels_out = [
        build_labels_for_task(task_ind, task) for task_ind, task in enumerate(tasks)
    ]

    labels_unshaped = list(zip(*labels_out))
    labels_shaped = []
    for ind in range(len(labels_unshaped)):
        if max_dims == 2:
            labels_shaped.append(np.concatenate(labels_unshaped[ind], axis=1))
        elif max_dims == 1:
            ## classification only
            labels_shaped.append(labels_unshaped[ind])
        else:
            raise Exception("This should not be possible that max_dims > 2.")
    return labels_shaped


def _build_labels_for_task(
    task: str,
    task_ind: int,
    output_mode: dict[str, str],
    result: BatchEncoding,
    labels: list,
    num_instances: int,
    max_length: int,
    label_lists: dict[str, list[str]],
    pad_classification: bool,
) -> Union[np.ndarray, list[np.ndarray]]:
    if output_mode[task] == tagging:
        return get_tagging_labels(task_ind, result, labels, num_instances)
    elif output_mode[task] == relex:
        return get_relex_labels(
            task,
            task_ind,
            result,
            labels,
            num_instances,
            max_length,
            label_lists,
        )
    elif output_mode[task] == classification:
        return get_classification_labels(
            task_ind,
            labels,
            num_instances,
            max_length,
            pad_classification,
        )


def get_tagging_labels(
    task_ind: int,
    result: BatchEncoding,
    labels: list,
    num_instances: int,
) -> list[np.ndarray]:
    encoded_labels = []
    for sent_ind in range(num_instances):
        word_ids = result["word_ids"][sent_ind]
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None or labels[sent_ind][task_ind] == [-100]:
                label_ids.append(-100)
                # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(labels[sent_ind][task_ind][word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
            else:
                # Dongfang's logic for beginning or interior of a word
                label_ids.append(-100)
            previous_word_idx = word_idx
        encoded_label = np.expand_dims(np.array(label_ids), 1).astype(int)
        encoded_labels.append(encoded_label)

    return encoded_labels


def get_relex_labels(
    task: str,
    task_ind: int,
    result: BatchEncoding,
    labels: list,
    num_instances: int,
    max_length: int,
    label_lists: dict[str, list[str]],
) -> list[np.ndarray]:
    encoded_labels = []
    # start by building a matrix that's N' x N' (word-piece length) with "None" as the default
    # for word pairs, and -100 (mask) as the default if one of word pair is a suffix token
    out_of_bounds = 0
    num_relations = 0

    def ids_getter(sent_ind: int) -> list[int]:
        return result["word_ids"][sent_ind]

    def relevant(word_idx: Union[int, None]) -> bool:
        return word_idx is not None

    for sent_ind in range(num_instances):
        word_ids = ids_getter(sent_ind)
        num_relations += len(labels[sent_ind][task_ind])
        wpi_to_tokeni = {}
        tokeni_to_wpi = {}
        sent_labels = np.zeros((max_length, max_length)) - 100

        ## align word-piece tokens to the tokenization we got as input and only assign labels to input tokens
        previous_word_idx = None
        for word_pos_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and relevant(word_idx):
                key = word_pos_idx
                val = len(wpi_to_tokeni)

                wpi_to_tokeni[key] = val
                tokeni_to_wpi[val] = key
            previous_word_idx = word_idx
        # make every label beween pairs a 0 to start:
        for wpi in wpi_to_tokeni.keys():
            for wpi2 in wpi_to_tokeni.keys():
                # leave the diagonals at -100 because you can't have a relation with itself and we
                # don't want to consider it because it may screw up the learning to have 2 such similar
                # tokens not involved in a relation.
                if wpi != wpi2:
                    sent_labels[wpi, wpi2] = label_lists[task].index("None")

        for label in labels[sent_ind][task_ind]:
            if label == "None":
                continue

            if label[0] not in tokeni_to_wpi or label[1] not in tokeni_to_wpi:
                out_of_bounds += 1
                continue

            wpi1 = tokeni_to_wpi[label[0]]
            wpi2 = tokeni_to_wpi[label[1]]

            sent_labels[wpi1][wpi2] = label[2]

        encoded_labels.append(sent_labels)
    if out_of_bounds > 0:
        logger.warning(
            "During relation processing,"
            f"there were {out_of_bounds} relations (out of {num_relations} total relations)"
            "where at least one argument was truncated so the relation could not be trained/predicted."
        )
    return encoded_labels


def get_classification_labels(
    task_ind: int,
    labels: list,
    num_instances: int,
    max_length: int,
    pad_classification: bool,
) -> np.ndarray:
    encoded_labels = []
    for inst_ind in range(num_instances):
        # if we try to combine classification with tagging/relex, we end up with non-rectangular label
        # arrays. so we need to pad out the classification target to be the length of the sequence
        # so that we can concatenate it. we'll have to account for this in the forward() and in the
        # compute metrics code as well.
        if pad_classification:
            padded_inst = np.zeros((max_length, 1)) - 100
            padded_inst[0] = labels[inst_ind][task_ind] if labels is not None else 0
            encoded_labels.append(padded_inst)
        else:
            encoded_labels.append(
                labels[inst_ind][task_ind] if labels is not None else 0
            )
    return np.array(encoded_labels)


# TODO: the following three functions seem to be unused in the codebase?


def truncate_features(feature: Union[InputFeatures, HierarchicalInputFeatures]):
    """
    Method to produce a truncated string representation of a feature.

    :param feature: the feature to represent
    :return: the truncated representation of the feature
    :meta private:
    """
    return (
        f"{feature.__class__.__name__}"
        "("
        f"input_ids={summarize(feature.input_ids)}, "
        f"attention_mask={summarize(feature.attention_mask)}, "
        f"token_type_ids={summarize(feature.token_type_ids)}, "
        f"event_tokens={summarize(feature.event_tokens)}, "
        f"label={summarize(feature.label)}"
        ")"
    )


def summarize(li) -> str:
    """
    Show a summarized version of a list. Used to reduce amount of text in logs for long input examples.
    :param li: Input list
    :return: Summary string
    :meta private:
    """
    if li is None:
        return "None"
    return str(truncate_list_of_lists(li)).replace('"', "").replace("'", "")


def truncate_list_of_lists(li: Union[list, str]) -> Union[list, str]:
    """
    For a list with more than 3 items, give the first item, summarize the middle items, and final item.
    If an element of the list is a list, it will recurse into that list and summarize that.
    Primarily used by :func:`summarize` to limit the amount of output in log files for really long input texts.
    :meta private:
    """
    if isinstance(li, str):
        return li
    if li:
        if len(li) > 3:
            li = [li[0], f"({len(li) - 2} more)", li[-1]]
        if isinstance(li[0], list):
            return [truncate_list_of_lists(item) for item in li]
        else:
            return li
    else:
        return li

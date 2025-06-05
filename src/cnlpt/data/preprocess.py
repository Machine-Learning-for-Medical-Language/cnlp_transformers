import logging
from collections.abc import Iterable
from typing import Any, Final, Union

import numpy as np
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .task_info import CLASSIFICATION, RELATIONS, TAGGING, TaskInfo

logger = logging.getLogger(__name__)
MISSING_DATA_STR: Final = "__None__"
MASK_VALUE: Final = -100


def preprocess_raw_data(
    batch: dict[str, Union[list[str], list[int], list[float]]],
    tokenizer: PreTrainedTokenizer,
    tasks: Union[Iterable[TaskInfo], None],
    max_length: Union[int, None] = None,
    inference_only: bool = False,
    hierarchical: bool = False,
    character_level: bool = False,
    chunk_len: int = -1,
    num_chunks: int = -1,
    insert_empty_chunk_at_beginning: bool = False,
) -> BatchEncoding:
    """Preprocess raw CNLP data for training/evaluation.

    Args:
        batch: A batch of data to preprocess.
        tokenizer: Tokenizer to use on the text in the dataset.
        tasks: A sequence of `TaskInfo` objects describing the tasks in the data. Optional.
        max_length: Maximum sequence length for the model. Optional, defaults to None.
        inference_only: Whether the data should be preprocessed for inference only. Defaults to False.
        hierarchical: Whether the data is being preprocessed for a hierarchical model. Defaults to False.
        character_level: Whether the model operates directly on characters instead of tokens. Defaults to False.
        chunk_len: The maximum length of a chunk (for hierarchical models). Defaults to -1.
        num_chunks: The number of chunks (for hierarchical models). Defaults to -1.
        insert_empty_chunk_at_beginning: Whether an empty chunk should be inserted at the beginning (for hierarchical models). Defaults to False.

    Raises:
        ValueError: If `inference_only` is True and `tasks` is None.

    Returns:
        The preprocessed batch.
    """
    if max_length is None:
        max_length = tokenizer.model_max_length

    tokenized_input = _tokenize_batch(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        hierarchical=hierarchical,
        character_level=character_level,
    )

    # Now that we have the labels for each instances, and we've tokenized the input sentences,
    # we need to solve the problem of aligning labels with word piece indexes for the tasks of tagging
    # (which has one label per pre-wordpiece token) and relations (which are defined as tuples which
    # contain pre-wordpiece token indices)
    if not inference_only:
        if tasks is None:
            raise ValueError("tasks must not be None if not in inference-only mode")

        labels = [
            _preprocess_raw_labels(raw=batch[task.name], task=task) for task in tasks
        ]

        # Convert the labels to column format that arrow prefers
        labels = list(zip(*labels))

        tokenized_input["label"] = _build_pytorch_labels(
            tokenized_input=tokenized_input,
            tasks=tasks,
            labels=labels,
            max_length=max_length,
        )

    if not character_level:
        tokenized_input["event_mask"] = _build_event_mask_word_piece(
            tokenized_input=tokenized_input,
            event_start_token_id=tokenizer.convert_tokens_to_ids("<e>"),
            event_end_token_id=tokenizer.convert_tokens_to_ids("</e>"),
        )
    else:
        tokenized_input["event_mask"] = _build_event_mask_character(
            tokenized_input=tokenized_input
        )
    if hierarchical:
        tokenized_input = _convert_features_to_hierarchical(
            tokenized_input,
            chunk_len=chunk_len,
            num_chunks=num_chunks,
            cls_id=tokenizer.cls_token_id,
            sep_id=tokenizer.sep_token_id,
            pad_id=tokenizer.pad_token_id,
            insert_empty_chunk_at_beginning=insert_empty_chunk_at_beginning,
        )

    return tokenized_input


def _convert_features_to_hierarchical(
    tokenized_input: BatchEncoding,
    chunk_len: int,
    num_chunks: int,
    cls_id: int,
    sep_id: int,
    pad_id: int,
    insert_empty_chunk_at_beginning: bool = False,
) -> BatchEncoding:
    """Chunk an instance of InputFeatures into an instance of HierarchicalInputFeatures
    for the hierarchical model.

    Args:
        features: The dictionary containing mappings from properties to lists of values for each instance for each of those properties
        chunk_len: The maximum length of a chunk
        num_chunks: The maximum number of chunks in the instance
        cls_id: The tokenizer's ID representing the CLS token
        sep_id: The tokenizer's ID representing the SEP token
        pad_id: The tokenizer's ID representing the PAD token
        insert_empty_chunk_at_beginning: Whether to insert an empty chunk at the beginning of the instance

    Returns:
        An instance of :class:`transformers.BatchEncoding` containing the chunked instance
    """

    for ind in range(len(tokenized_input.input_ids)):
        # Get feature variables
        # input_ids_, attention_mask_, token_type_ids_, event_tokens_, label_ = astuple(features)
        input_ids_ = tokenized_input.input_ids[ind]
        attention_mask_ = tokenized_input["attention_mask"][ind]
        token_type_ids_ = tokenized_input.get("token_type_ids", None)
        if token_type_ids_ is not None:
            token_type_ids_ = token_type_ids_[ind]
        event_tokens_ = tokenized_input["event_mask"][ind]

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
            formatted_chunk = [cls_type, *chunk, sep_type]
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
                assert token_type_ids_ is not None
                chunks_token_type_ids.append(
                    format_chunk(token_type_ids_[start:end], cls_type=0, sep_type=0)
                )
            if chunks_event_tokens is not None:
                chunks_event_tokens.append(
                    format_chunk(event_tokens_[start:end], cls_type=1, sep_type=1)
                )

            start = end

        def create_pad_chunk(cls_type=cls_id, sep_type=sep_id, pad_type=pad_id):
            return pad_chunk([cls_type, sep_type], pad_type=pad_type)

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

        tokenized_input.input_ids[ind] = chunks
        tokenized_input["attention_mask"][ind] = chunks_attention_mask
        if token_type_ids_ is not None:
            tokenized_input["token_type_ids"][ind] = chunks_token_type_ids
        tokenized_input["event_mask"][ind] = chunks_event_tokens

    return tokenized_input


def _get_word_ids(
    tokenizer: PreTrainedTokenizer,
    tokenized_input: BatchEncoding,
    character_level: bool,
) -> list[list[Union[int, None]]]:
    if tokenizer.is_fast:
        return [
            tokenized_input.word_ids(i) for i in range(len(tokenized_input.input_ids))
        ]
    elif character_level:
        # slow tokenizers -> build your own word ids
        special_token_ids = set(
            [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.cls_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
                tokenizer.unk_token_id,
            ]
        )

        def get_word_ids(indices: Iterable[int]) -> list[Union[int, None]]:
            current = 0
            raw: list[Union[int, None]] = []
            for index in indices:
                if index in special_token_ids:
                    raw.append(None)
                else:
                    raw.append(current)
                    current += 1
            return raw

        return [get_word_ids(indices) for indices in tokenized_input.input_ids]
    else:
        raise ValueError(
            f"{type(tokenizer).__name__}"
            "is a slow ( non-Rust ) tokenizer and thus word_ids is not implemented by default, "
            "you can provide your own implementation for extracting word_ids "
            "(see  https://huggingface.co/docs/tokenizers/main/en/api/encoding#tokenizers.Encoding.word_ids) for "
            "your model in this file"
        )


def _tokenize_batch(
    batch: dict[str, Union[list[str], list[int], list[float]]],
    tokenizer: PreTrainedTokenizer,
    max_length: Union[int, None],
    hierarchical: bool,
    character_level: bool,
) -> BatchEncoding:
    # Try to infer the structure based on column names
    column_names = batch.keys()
    if "text" in column_names:
        if character_level:
            sentences = [str(inst) for inst in batch["text"]]
        else:
            sentences = [str(inst).split(" ") for inst in batch["text"]]
    elif "text_a" in column_names and "text_b" in column_names:
        # FIXME - join these columns on SEP token
        raise NotImplementedError(
            "2-sentence classification has not been re-implemented yet."
        )
    else:
        raise ValueError(
            'The data does not seem to have a text column (literally a column labeled "text" is required)'
        )

    if hierarchical:
        padding = False
    else:
        padding = "max_length"

    tokenized_batch = tokenizer(
        sentences,
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=not character_level,
    )

    tokenized_batch["word_ids"] = _get_word_ids(
        tokenizer=tokenizer,
        tokenized_input=tokenized_batch,
        character_level=character_level,
    )

    return tokenized_batch


def _preprocess_raw_labels(
    raw: Union[list[str], list[int], list[float]], task: TaskInfo
):
    mask_missing: Final = {MISSING_DATA_STR: MASK_VALUE}
    if task.type == CLASSIFICATION:
        # labels is just a list of one label for each instance
        return [task.get_label_id(str(label), specials=mask_missing) for label in raw]
    elif task.type == TAGGING:
        # labels is a list of lists, where each internal list is the set of tags for that instance.
        return [
            [
                task.get_label_id(str(tag), specials=mask_missing)
                for tag in str(tags).split(" ")
            ]
            for tags in raw
        ]
    elif task.type == RELATIONS:
        preprocessed: list[Union[list[str], list[tuple[int, int, int]]]] = []
        for relations in raw:
            if relations in (None, "None"):
                preprocessed.append(["None"])
            else:
                # The label for a sentence with multiple relations looks like this:
                # (105,109,OVERLAP) , (64,66,CONTAINS) , (100,105,CONTAINS)
                # Split into relations, then remove parens and split with commas
                # into relation components (start offset, end offset, category)
                inst_labels: list[tuple[int, int, int]] = []
                for rel in str(relations).split(" , "):
                    start_token, end_token, category = rel[1:-1].split(",")
                    inst_labels.append(
                        (
                            int(start_token),
                            int(end_token),
                            task.get_label_id(category),
                        )
                    )
                preprocessed.append(inst_labels)
        return preprocessed
    else:
        raise NotImplementedError(
            f"This method is not complete for output mode {task.type}"
        )


def _build_event_mask_word_piece(
    tokenized_input: BatchEncoding, event_start_token_id: int, event_end_token_id: int
):
    """Create arrays corresponding to input tokens where the events to be classified contain special mask tokens.
    These are used if the --event flag is specified to classify event tokens rather than the [CLS] token.

    Args:
        tokenized_input: The input encodings of the tokens
        event_start_token_id: The special token index used to indicate the start of an event.
        event_end_token_id: The special token index used to indicate the end of an event.

    Returns:
        The list of lists of per-instance event mask values corresponding to the input tokens.
    """
    event_tokens: list[list[int]] = []
    input_ids: list[int]
    for input_ids in tokenized_input.input_ids:
        try:
            event_start = input_ids.index(event_start_token_id)
        except ValueError:
            event_start = -1

        try:
            event_end = input_ids.index(event_end_token_id)
        except ValueError:
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


def _build_event_mask_character(tokenized_input: BatchEncoding):
    event_tokens: list[list[int]] = []
    input_ids: list[int]
    for input_ids in tokenized_input.input_ids:
        inst_event_tokens = [1] * len(input_ids)
        event_tokens.append(inst_event_tokens)

    return event_tokens


def _build_pytorch_labels(
    tokenized_input: BatchEncoding,
    tasks: Iterable[TaskInfo],
    labels: list[tuple[Any, ...]],
    max_length: int,
) -> list[np.ndarray]:
    # labels_out = []
    # TODO -- also adapt to character level
    pad_classification = False
    task_types = {task.type for task in tasks}
    if RELATIONS in task_types or TAGGING in task_types:
        # we have tagging as the highest dimensional output
        max_dims = 2
        if CLASSIFICATION in task_types:
            pad_classification = True
    else:
        # classification only
        max_dims = 1

    labels_out = [
        _build_labels_for_task(
            task=task,
            tokenized_input=tokenized_input,
            labels=labels,
            max_length=max_length,
            pad_classification=pad_classification,
        )
        for task in tasks
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
    task: TaskInfo,
    tokenized_input: BatchEncoding,
    labels: list[tuple[Any, ...]],
    max_length: int,
    pad_classification: bool,
) -> Union[np.ndarray, list[np.ndarray]]:
    if task.type == TAGGING:
        return _get_tagging_labels(task, tokenized_input, labels)
    elif task.type == RELATIONS:
        return _get_relex_labels(task, tokenized_input, labels, max_length)
    elif task.type == CLASSIFICATION:
        return _get_classification_labels(task, labels, max_length, pad_classification)
    raise ValueError(f"unsupported task type {task.type}")


def _get_tagging_labels(
    task: TaskInfo,
    tokenized_input: BatchEncoding,
    labels: list[tuple[Any, ...]],
) -> list[np.ndarray]:
    encoded_labels: list[np.ndarray] = []
    for tags, word_ids in zip(
        (row[task.index] for row in labels),
        tokenized_input["word_ids"],
    ):
        previous_word_idx = None
        label_ids: list[int] = []
        for word_idx in word_ids:
            if word_idx is None or tags == [MASK_VALUE]:
                # Special tokens have a word id that is None. We set the label to MASK_VALUE so they are automatically
                # ignored in the loss function.
                label_ids.append(MASK_VALUE)
            elif word_idx != previous_word_idx:
                # We set the label for the first token of each word.
                label_ids.append(tags[word_idx])
            else:
                # Dongfang's logic for beginning or interior of a word
                label_ids.append(MASK_VALUE)
            previous_word_idx = word_idx
        encoded_label = np.expand_dims(np.array(label_ids), 1).astype(int)
        encoded_labels.append(encoded_label)

    return encoded_labels


def _get_relex_labels(
    task: TaskInfo,
    tokenized_input: BatchEncoding,
    labels: list[tuple[Any, ...]],
    max_length: int,
) -> list[np.ndarray]:
    encoded_labels: list[np.ndarray] = []
    # start by building a matrix that's N' x N' (word-piece length) with "None" as the default
    # for word pairs, and MASK_VALUE as the default if one of word pair is a suffix token
    out_of_bounds = 0
    num_relations = 0

    for relations, word_ids in zip(
        (row[task.index] for row in labels),
        tokenized_input["word_ids"],
    ):
        num_relations += len(relations)
        wpi_to_tokeni = {}
        tokeni_to_wpi = {}
        sent_labels = np.ones((max_length, max_length)) * MASK_VALUE

        ## align word-piece tokens to the tokenization we got as input and only assign labels to input tokens
        previous_word_idx = None
        for word_pos_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and word_idx is not None:
                key = word_pos_idx
                val = len(wpi_to_tokeni)

                wpi_to_tokeni[key] = val
                tokeni_to_wpi[val] = key
            previous_word_idx = word_idx
        # make every label beween pairs a 0 to start:
        for wpi in wpi_to_tokeni.keys():
            for wpi2 in wpi_to_tokeni.keys():
                # leave the diagonals at MASK_VALUE because you can't have a relation with itself and we
                # don't want to consider it because it may screw up the learning to have 2 such similar
                # tokens not involved in a relation.
                if wpi != wpi2:
                    sent_labels[wpi, wpi2] = task.get_label_id("None")

        for tup_or_none in relations:
            if tup_or_none == "None":
                continue
            start, end, val = tup_or_none
            if start not in tokeni_to_wpi or end not in tokeni_to_wpi:
                out_of_bounds += 1
                continue

            wpi1 = tokeni_to_wpi[start]
            wpi2 = tokeni_to_wpi[end]

            sent_labels[wpi1][wpi2] = val

        encoded_labels.append(sent_labels)
    if out_of_bounds > 0:
        logger.warning(
            "During relation processing,"
            f"there were {out_of_bounds} relations (out of {num_relations} total relations)"
            "where at least one argument was truncated so the relation could not be trained/predicted."
        )
    return encoded_labels


def _get_classification_labels(
    task: TaskInfo,
    labels: list[tuple[Any, ...]],
    max_length: int,
    pad_classification: bool,
) -> np.ndarray:
    encoded_labels = []
    for label in [row[task.index] for row in labels]:
        # if we try to combine classification with tagging/relex, we end up with non-rectangular label
        # arrays. so we need to pad out the classification target to be the length of the sequence
        # so that we can concatenate it. we'll have to account for this in the forward() and in the
        # compute metrics code as well.
        if pad_classification:
            padded_inst = np.ones((max_length, 1)) * MASK_VALUE
            padded_inst[0] = label
            encoded_labels.append(padded_inst)
        else:
            encoded_labels.append(label)
    return np.array(encoded_labels)

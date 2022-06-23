import os
from os.path import basename, dirname
import time
import logging
import json

from filelock import FileLock
from typing import Callable, Dict, Optional, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass, field, asdict, astuple
from enum import Enum

from .cnlp_processors import cnlp_processors, cnlp_output_modes, classification, tagging, relex, mtl

special_tokens = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']

logger = logging.getLogger(__name__)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

class Split(Enum):
    """
    Enum representing the three data splits for model development.
    """
    train = "train"
    dev = "dev"
    test = "test"

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        event_tokens: (Optional)
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    event_tokens: Optional[List[int]] = None
    label: List[Optional[Union[int, float, List[int], List[Tuple[int]]]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self)) + "\n"


@dataclass(frozen=True)
class HierarchicalInputFeatures:
    """
    A single set of features of data for the hierarchical model.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        event_tokens: (Optional)
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]] = None
    token_type_ids: Optional[List[List[int]]] = None
    event_tokens: Optional[List[List[int]]] = None
    label: List[Optional[Union[int, float, List[int], List[Tuple[int]]]]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self)) + "\n"


def cnlp_convert_features_to_hierarchical(
        features: InputFeatures,
        chunk_len: int,
        num_chunks: int,
        cls_id: int,
        sep_id: int,
        pad_id: int,
        insert_empty_chunk_at_beginning: bool = False,
        # cls_token_at_end=False,
        # sequence_a_segment_id=0,
        # cls_token_segment_id=0,
        # pad_token_segment_id=0,
        # use_special_token=True,
) -> HierarchicalInputFeatures:
    """
    Chunk an instance of InputFeatures into an instance of HierarchicalInputFeatures
    for the hierarchical model.

    :param InputFeatures features: the old instance
    :param int chunk_len: the maximum length of a chunk
    :param int num_chunks: the maximum number of chunks in the instance
    :param int cls_id: the tokenizer's ID representing the CLS token
    :param int sep_id: the tokenizer's ID representing the SEP token
    :param int pad_id: the tokenizer's ID representing the PAD token
    :param bool insert_empty_chunk_at_beginning: whether to insert an
        empty chunk at the beginning of the instance
    :rtype: HierarchicalInputFeatures
    :return: an instance of `HierarchicalInputFeatures` containing the chunked instance
    """
    # Get feature variables
    input_ids_, attention_mask_, token_type_ids_, event_tokens_, label_ = astuple(features)

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

    def format_chunk(chunk, cls_type=cls_id, sep_type=sep_id, pad_type=pad_id, pad=True):
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
        end = min(
            start + chunk_len - 2,
            len(input_ids_)
        )

        # if we are ending on a PAD token or the SEP token, end before the SEP token
        if input_ids_[end-1] in {pad_id, sep_id} and sep_id in input_ids_[start:end]:
            end = input_ids_.index(sep_id, start, end)

        chunks.append(format_chunk(input_ids_[start:end]))
        if chunks_attention_mask is not None:
            chunks_attention_mask.append(format_chunk(attention_mask_[start:end], cls_type=1, sep_type=1))
        if chunks_token_type_ids is not None:
            chunks_token_type_ids.append(format_chunk(token_type_ids_[start:end], cls_type=0, sep_type=0))
        if chunks_event_tokens is not None:
            chunks_event_tokens.append(format_chunk(event_tokens_[start:end], cls_type=1, sep_type=1))

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

    return HierarchicalInputFeatures(chunks, chunks_attention_mask, chunks_token_type_ids, chunks_event_tokens, label_)


def cnlp_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task: str = None,
    label_list: Optional[List[str]] = None,
    output_mode: Optional[str] = None,
    inference: bool = False,
    hierarchical: bool = False,
    chunk_len: int = -1,
    num_chunks: int = -1,
    insert_empty_chunk_at_beginning: bool = False,
    truncate_examples: bool = False,
) -> Union[List[InputFeatures], List[HierarchicalInputFeatures]]:
    """
    Processes the list of :class:`transformers.InputExample` generated by
    the processor defined in :data:`cnlpt.cnlp_processors.cnlp_processors`
    and converts the examples into a list of :class:`InputFeatures` or
    :class:`HierarchicalInputFeatures`, depending on the model.

    :param typing.List[transformers.data.processors.utils.InputExample] examples:
        the list of examples to convert
    :param transformers.tokenization_utils.PreTrainedTokenizer tokenizer: the tokenizer
    :param typing.Optional[int] max_length: the maximum sequence length
        at which to truncate examples
    :param str task: the task name
    :param typing.Optional[typing.List[str]] label_list: the list of labels
        for this task. If not provided explicitly, it will be retrieved from
        the processor with :meth:`transformers.DataProcessor.get_labels`.
    :param typing.Optional[str] output_mode: the output mode for this task.
        If not provided explicitly, it will be retrieved from
        :data:`cnlpt.cnlp_processors.cnlp_output_modes`.
    :param bool inference: whether we're doing training or inference only -- if inference mode the labels associated with examples can't be trusted.
    :param bool hierarchical: whether to structure the data for the hierarchical
        model (:class:`cnlpt.HierarchicalTransformer.HierarchicalModel`)
    :param int chunk_len: for the hierarchical model, the length of each
        chunk in tokens
    :param int num_chunks: for the hierarchical model, the number of chunks
    :param bool insert_empty_chunk_at_beginning: for the hierarchical model,
        whether to insert an empty chunk at the beginning of the list of chunks
        (equivalent in theory to a CLS chunk).
    :param bool truncate_examples: whether to truncate the string representation
        of the example instances printed to the log
    :rtype: typing.Union[typing.List[InputFeatures], typing.List[HierarchicalInputFeatures]]
    :return: the list of converted input features
    """
    event_start_ind = tokenizer.convert_tokens_to_ids('<e>')
    event_end_ind = tokenizer.convert_tokens_to_ids('</e>')
    
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = cnlp_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = cnlp_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            # give it a random label, if we didn't specify a label with the data we won't be comparing it.
            # return list(label_map.values())[0]
            return None
        if output_mode == classification:
            try:
                return label_map[example.label]
            except:
                logger.error('Error with example %s' % (example.guid))
                raise Exception()

        elif output_mode == "regression":
            return float(example.label)
        elif output_mode == tagging:
            return [ label_map[label] for label in example.label]
        elif output_mode == relex:
            return [ (int(start_token),int(end_token),label_map.get(category, 0)) for (start_token,end_token,category) in example.label]
        elif output_mode == mtl:
            return [ label_map[x] for x in example.label]

        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    if examples[0].text_b is None:
        sentences = [example.text_a.split(' ') for example in examples]
    else:
        sentences = [(example.text_a, example.text_b) for example in examples]

    batch_encoding = tokenizer(
        sentences,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
    )

    # This code has to solve the problem of properly setting labels for word pieces that do not actually need to be tagged.
    if not inference:
        encoded_labels = []
        if output_mode == tagging:
            for sent_ind,sent in enumerate(sentences):
                sent_labels = []

                ## align word-piece tokens to the tokenization we got as input and only assign labels to input tokens
                word_ids = batch_encoding.word_ids(batch_index=sent_ind)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[sent_ind][word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx

                encoded_labels.append(np.array(label_ids))

            labels = encoded_labels
        elif output_mode == relex:
            # start by building a matrix that's N' x N' (word-piece length) with "None" as the default
            # for word pairs, and -100 (mask) as the default if one of word pair is a suffix token
            out_of_bounds = 0
            num_relations = 0
            for sent_ind, sent in enumerate(sentences):
                word_ids = batch_encoding.word_ids(batch_index=sent_ind)
                num_relations += len(labels[sent_ind])
                wpi_to_tokeni = {}
                tokeni_to_wpi = {}
                sent_labels = np.zeros( (max_length, max_length)) - 100

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
                        # leave the diagonals at -100 because you can't have a relation with itself and we
                        # don't want to consider it because it may screw up the learning to have 2 such similar
                        # tokens not involved in a relation.
                        if wpi != wpi2:
                            sent_labels[wpi,wpi2] = 0.0

                for label in labels[sent_ind]:
                    if not label[0] in tokeni_to_wpi or not label[1] in tokeni_to_wpi:
                        out_of_bounds +=1
                        continue

                    wpi1 = tokeni_to_wpi[label[0]]
                    wpi2 = tokeni_to_wpi[label[1]]

                    sent_labels[wpi1][wpi2] = label[2]

                encoded_labels.append(sent_labels)
            labels = encoded_labels
            if out_of_bounds > 0:
                logging.warn('During relation processing, there were %d relations (out of %d total relations) where at least one argument was truncated so the relation could not be trained/predicted.' % (out_of_bounds, num_relations) )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        try:
            event_start = inputs['input_ids'].index(event_start_ind)
        except:
            event_start = -1

        try:
            event_end = inputs['input_ids'].index(event_end_ind)
        except:
            event_end = len(inputs['input_ids'])-1

        inputs['event_tokens'] = [0] * len(inputs['input_ids'])
        if event_start >= 0:
            inputs['event_tokens'] = [0] * event_start + [1] * (event_end-event_start+1) + [0] * (len(inputs['input_ids'])-event_end-1)
        else:
            inputs['event_tokens'] = [1] * len(inputs['input_ids'])

        if inference:
            label = None
        else:
            label = [labels[i]]
        feature = InputFeatures(**inputs, label=label)
        if hierarchical:
            feature = cnlp_convert_features_to_hierarchical(
                feature,
                chunk_len=chunk_len,
                num_chunks=num_chunks,
                cls_id=tokenizer.cls_token_id,
                sep_id=tokenizer.sep_token_id,
                pad_id=tokenizer.pad_token_id,
                insert_empty_chunk_at_beginning=insert_empty_chunk_at_beginning,
            )
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % truncate_features(features[i]) if truncate_examples else features[i])

    return features


def truncate_features(feature: Union[InputFeatures, HierarchicalInputFeatures]):
    """
    Method to produce a truncated string representation of a feature.

    :param typing.Union[InputFeatures, HierarchicalInputFeatures] feature:
        the feature to represent
    :rtype: str
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


def summarize(li):
    if li is None:
        return 'None'
    return str(truncate_list_of_lists(li)).replace('"', '').replace("'", '')


def truncate_list_of_lists(li: Union[list, str]) -> Union[list, str]:
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using :class:`transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: List[str] = field(
        metadata={"help": "The input data dirs. A space-separated list of directories that "
                          "should contain the .tsv files (or other data files) for the task. "
                          "Should be presented in the same order as the task names."}
    )

    task_name: List[str] = field(default_factory=lambda: None, metadata={
        "help": "A space-separated list of tasks to train on: " + ", ".join(cnlp_processors.keys())
    })
    # field(
        
    #     metadata={"help": "A space-separated list of tasks to train on: " + ", ".join(cnlp_processors.keys())})

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    weight_classes: bool = field(
        default=False, metadata={"help": "A flag that indicates whether class-specific loss should be used. "
                                         "This can be useful in cases with severe class imbalance. The formula "
                                         "for a weight of a class is the count of that class divided the count "
                                         "of the rarest class."}
    )

    chunk_len: Optional[int] = field(default=None, metadata={"help": "Chunk length for hierarchical model"})

    num_chunks: Optional[int] = field(default=None, metadata={"help": "Max chunk count for hierarchical model"})

    insert_empty_chunk_at_beginning: bool = field(default=False, metadata={
        "help": "Whether to insert an empty chunk for hierarchical model"
    })

    truncate_examples: bool = field(default=False, metadata={
        "help": "Whether to truncate input examples when displaying them in the log"
    })


class ClinicalNlpDataset(Dataset):
    """
    Copy-pasted from GlueDataset with glue task-specific code changed;
    moved into here to be self-contained.

    :param DataTrainingArguments args: the data training args for this experiment
    :param transformers.tokenization_utils.PreTrainedTokenizer tokenizer: the tokenizer
    :param typing.Optional[int] limit_length: if provided, the number of
        examples to include in the dataset
    :param typing.Union[str, Split] mode: the data split mode of this dataset
        (:obj:`"train"`, :obj:`"dev"`, :obj:`"test"`)
    :param typing.Optional[str] cache_dir: if provided, the directory to save/load a cache
        of this dataset
    :param bool hierarchical: whether to structure the data for the hierarchical
        model (:class:`cnlpt.HierarchicalTransformer.HierarchicalModel`)
    """
    args: DataTrainingArguments
    output_mode: List[str]
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
        hierarchical: bool = False,
    ):
        self.args = args
        self.processors = []
        self.output_mode = []
        self.class_weights = []
        self.hierarchical = hierarchical

        for task in args.task_name:
            self.processors.append(cnlp_processors[task]())
            self.output_mode.append(cnlp_output_modes[task])
            if self.output_mode[-1] == mtl:
                for subtask in range(self.processors[-1].get_num_tasks()):
                    self.class_weights.append(None)
            else:
                self.class_weights.append(None)
        
        self.features = None

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        # Load data features from cache or dataset file
        self.label_lists = [processor.get_labels() for processor in self.processors]

        for task_ind,data_dir in enumerate(args.data_dir):
            datadir = dirname(data_dir) if data_dir[-1] == '/' else data_dir
            domain = basename(datadir)
            dataconfig = basename(dirname(datadir))
            num_subtasks = self.processors[task_ind].get_num_tasks()

            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else data_dir,
                "cached_{}-{}_{}_{}_{}".format(
                    dataconfig, domain, mode.value, tokenizer.__class__.__name__, str(args.max_seq_length),
                ),
            )
            if self.hierarchical:
                cached_features_file += '_hier'

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not args.overwrite_cache:
                    start = time.time()
                    features = torch.load(cached_features_file)
                    logger.info(
                        f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                    )
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")

                    if mode == Split.dev:
                        examples = self.processors[task_ind].get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = self.processors[task_ind].get_test_examples(data_dir)
                    else:
                        examples = self.processors[task_ind].get_train_examples(data_dir)
                    if limit_length is not None:
                        examples = examples[:limit_length]
                    features = cnlp_convert_examples_to_features(
                        examples,
                        tokenizer,
                        max_length=args.max_seq_length,
                        label_list=self.label_lists[task_ind],
                        output_mode=self.output_mode[task_ind],
                        inference=mode == Split.test,
                        hierarchical=self.hierarchical,
                        chunk_len=self.args.chunk_len,
                        num_chunks=self.args.num_chunks,
                        insert_empty_chunk_at_beginning=self.args.insert_empty_chunk_at_beginning,
                        truncate_examples=self.args.truncate_examples,
                    )
                    start = time.time()
                    torch.save(features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

                if self.args.weight_classes and mode == Split.train:
                    if num_subtasks == 1:
                        class_counts = [0] * len(self.label_lists[task_ind])
                        for feature in features:
                            labels = feature.label[0]
                            vals, counts = np.unique(labels, return_counts=True)
                            for val_ind,val in enumerate(vals):
                                if val >= 0:
                                    class_counts[int(val)] += counts[val_ind]

                        self.class_weights[task_ind] = min(class_counts) / class_counts
                    else:
                        class_counts = np.zeros( (num_subtasks, len(self.label_lists[task_ind])) )
                        for feature in features:
                            labels = feature.label[0]
                            for subtask_ind,label in enumerate(labels):
                                class_counts[subtask_ind][label] += 1
                                
                        self.class_weights[subtask_ind] = min(class_counts[subtask_ind]) / class_counts[subtask_ind]


                if self.features is None:
                    self.features = features
                else:
                    assert len(features) == len(self.features)
                    if self.features[0].label is None:
                        assert features[0].label is None, 'Some of the tasks have None labels and others do not, they should be consistent!'
                    else:
                        # we should have all non-label features be the same, so we can essentially discard subsequent
                        # datasets input features. So we'll append the labels from that features list and discard the duplicate input features.
                        for feature_ind,feature in enumerate(features):
                            if len(self.features[feature_ind].label[0].shape) == 1 and len(feature.label[0].shape) == 1:
                                self.features[feature_ind].label[0] = np.stack([self.features[feature_ind].label[0],
                                                                            feature.label[0]])
                            else:
                                self.features[feature_ind].label[0] = np.concatenate([self.features[feature_ind].label[0],
                                                                            feature.label[0]])

    def __len__(self) -> int:
        """
        Length method for this class.

        :rtype: int
        :return: the number of instances in the dataset
        """
        return len(self.features)

    def __getitem__(self, i):
        """
        Getitem method for this class.

        :param i: the index of the example to retrieve
        :rtype: typing.Union[InputFeatures, HierarchicalInputFeatures]
        :return: the example at index `i`
        """
        return self.features[i]

    def get_labels(self):
        """
        Retrieve the label lists for all the tasks for the dataset.

        :rtype: typing.List[typing.List[str]]
        :return: the list of label lists
        """
        return self.label_lists

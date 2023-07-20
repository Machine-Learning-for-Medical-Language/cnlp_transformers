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
from transformers import BatchEncoding, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer
from datasets import Features
from dataclasses import dataclass, field, asdict, astuple
import datasets
from enum import Enum

from .cnlp_processors import classification, tagging, relex, mtl, AutoProcessor

special_tokens = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']
text_columns = ['text', 'text_a', 'text_b']
none_column = '__None__'

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
        features: BatchEncoding,
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

    for ind in range(len(features['input_ids'])):
        # Get feature variables
        # input_ids_, attention_mask_, token_type_ids_, event_tokens_, label_ = astuple(features)
        input_ids_ = features['input_ids'][ind]
        attention_mask_ = features['attention_mask'][ind]
        token_type_ids_ = features.get('token_type_ids', None)
        if not token_type_ids_ is None:
            token_type_ids_ = token_type_ids_[ind]
        event_tokens_ = features['event_mask'][ind]

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

        features['input_ids'][ind] = chunks
        features['attention_mask'][ind] = chunks_attention_mask
        if not token_type_ids_ is None:
            features['token_type_ids'][ind] = chunks_token_type_ids
        features['event_mask'][ind] = chunks_event_tokens

    return features


def cnlp_preprocess_data(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    tasks: List[str] = None,
    label_lists: Optional[Dict[str,List[str]]] = None,
    output_modes: Optional[Dict[str,str]] = None,
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

    :param examples:
        the list of examples to convert
    :param tokenizer: the tokenizer
    :param max_length: the maximum sequence length
        at which to truncate examples
    :param tasks: the task name(s) in a list, used to index the labels in the examples list.
    :param label_list: a mapping from
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
    
    if max_length is None:
        max_length = tokenizer.max_len

    # Try to infer the structure based on column names
    if 'text' in examples.keys():
        sentences = [example.split(' ') for example in examples['text']]
        num_instances = len(examples['text'])
    elif 'text_b' in examples.keys():
        # FIXME - not sure if this is right but doesn't get used much in our data
        raise NotImplementedError("2-sentence classification has not been re-implemented yet.")
        sentences = (examples['text_a'], examples['text_b'])        
    else:
        raise Exception('The data does not seem to have a text column (literally a column labeled "text" is required)')
    
    if hierarchical:
        padding = False
    else:
        padding = 'max_length'

    result = tokenizer(
        sentences,
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )

    # Now that we have the labels for each instances, and we've tokenized the input sentences, 
    # we need to solve the problem of aligning labels with word piece indexes for the tasks of tagging
    # (which has one label per pre-wordpiece token) and relations (which are defined as tuples which
    # contain pre-wordpiece token indices)
    if not inference:
        # Create a label map for each task in this dataset: { task1 => {label_0: 0, label_1: 1, label_2:, 2}, task2 => {label_0: 0, label_1:1} }
        label_map = {task: {label: i for i, label in enumerate(label_lists[task])} for task in label_lists.keys()}
        for task in tasks:
            if none_column in label_map[task]:
                raise Exception("There is a column named %s which is a reserved name" % (none_column))
            label_map[task][none_column] = -100

        raw_labels = []
        labels = []

        # Create a list of mapped labels for every task in this dataset, with different mapping tactics for different types,
        # classification vs tagging vs. relations.
        for task_ind,task in enumerate(tasks):
            task_labels = []
            raw_labels.append(examples[task])

            if output_modes[task] == classification:
                task_labels = [label_map[task][label] for label in raw_labels[task_ind]]
                # labels is just a list of one label for each instance
            elif output_modes[task] == tagging:
                task_labels = [ [label_map[task][label] for label in inst_labels.split()] for inst_labels in raw_labels[task_ind]]
                # labels is a list of lists, where each internal list is the set of tags for that instance.
            elif output_modes[task] == relex:
                for inst_rels in raw_labels[task_ind]:
                    if inst_rels is None or inst_rels == 'None':
                        task_labels.append(['None'])
                    else:
                        # The label for a sentence with multiple relations looks like this:
                        # (105,109,OVERLAP) , (64,66,CONTAINS) , (100,105,CONTAINS) , (81,88,CONTAINS) , (81,95,CONTAINS) , (105,106,OVERLAP) , (81,100,CONTAINS)
                        # Split into relations, then remove parens and split with commas into relation components (start offset, end offset, category)
                        inst_labels = []
                        for rel in inst_rels.split(' , '):
                            start_token, end_token, category = rel[1:-1].split(',')
                            inst_labels.append( (int(start_token), int(end_token), label_map[task].get(category, 0)))
                        task_labels.append(inst_labels)
            else:
                raise NotImplementedError('This method is not complete for output mode %s' % (output_modes[task],) )
            labels.append(task_labels)

        # Convert the labels to column format that arrow prefers
        labels = list(zip(*labels))

        result['label'] = _build_pytorch_labels(result, tasks, labels, output_modes, num_instances, max_length, label_lists)
    # else:
        # result['label'] =  [ (0,) for i in range(num_instances)]

    result['event_mask'] = _build_event_mask(result, 
                                            num_instances,
                                            tokenizer.convert_tokens_to_ids('<e>'),
                                            tokenizer.convert_tokens_to_ids('</e>'))

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

def _build_pytorch_labels(result:BatchEncoding, tasks:List[str], labels:List, output_modes:Dict[str,str], num_instances:int, max_length:int, label_lists: List[List[str]]):
    '''
    _build_pytorch_labels: we do two things here: map from labels in input space to ints in a softmax, and in a data
    structure that can contain multiple task types such that the Trainer class will be happy with, and then that
    the model.forward() can unpack easily to do the loss calculations.
    '''
    labels_out = []

    pad_classification = False
    if relex in output_modes.values() or tagging in output_modes.values():
        # we have tagging as the highest dimensional output
        max_dims = 2
        if classification in output_modes.values():
            pad_classification = True
    else:
        # classification only
        max_dims = 1

    for task_ind, task in enumerate(tasks):
        encoded_labels = []
        if output_modes[task] == tagging:
            for sent_ind in range(num_instances):
                sent_labels = []

                ## align word-piece tokens to the tokenization we got as input and only assign labels to input tokens
                word_ids = result.word_ids(batch_index=sent_ind)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function. Non-tagging tasks will be labeled as [-100], all label ids are -100
                    if word_idx is None or labels[sent_ind][task_ind] == [-100]:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(labels[sent_ind][task_ind][word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx

                encoded_labels.append(np.expand_dims(np.array(label_ids), 1))

            labels_out.append(encoded_labels)
        elif output_modes[task] == relex:
            # start by building a matrix that's N' x N' (word-piece length) with "None" as the default
            # for word pairs, and -100 (mask) as the default if one of word pair is a suffix token
            out_of_bounds = 0
            num_relations = 0
            for sent_ind in range(num_instances):
                word_ids = result.word_ids(batch_index=sent_ind)
                num_relations += len(labels[sent_ind][task_ind])
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
                            sent_labels[wpi,wpi2] = label_lists[task].index('None')

                for label in labels[sent_ind][task_ind]:
                    if label == "None":
                        continue
                    
                    if not label[0] in tokeni_to_wpi or not label[1] in tokeni_to_wpi:
                        out_of_bounds +=1
                        continue
                    

                    wpi1 = tokeni_to_wpi[label[0]]
                    wpi2 = tokeni_to_wpi[label[1]]

                    sent_labels[wpi1][wpi2] = label[2]

                encoded_labels.append(sent_labels)
            labels_out.append(encoded_labels)
            if out_of_bounds > 0:
                logger.warn(
                    'During relation processing, there were %d relations (out of %d total relations) where at least one argument was truncated so the relation could not be trained/predicted.' % (out_of_bounds, num_relations)
                )
                
        elif output_modes[task] == classification:
            for inst_ind in range(num_instances):
                # if we try to combine classification with tagging/relex, we end up with non-rectangular label
                # arrays. so we need to pad out the classification target to be the length of the sequence
                # so that we can concatenate it. we'll have to account for this in the forward() and in the
                # compute metrics code as well.
                if pad_classification:
                    padded_inst = np.zeros( (max_length,1) ) - 100
                    padded_inst[0] = labels[inst_ind][task_ind]
                    encoded_labels.append( padded_inst )
                else:
                    encoded_labels.append(labels[inst_ind][task_ind])
            labels_out.append(np.array(encoded_labels))
    
    labels_unshaped =  list(zip(*labels_out))
    labels_shaped = []
    
    for ind in range(len(labels_unshaped)):
        if max_dims == 2:
            ## relations or tagging and possibly classification too
            labels_shaped.append( np.concatenate( labels_unshaped[ind], axis=1 ) )
        elif max_dims == 1:
            ## classification only
            labels_shaped.append( labels_unshaped[ind] )
        else:
            raise Exception('This should not be possible that max_dims > 2.')
    
    return labels_shaped

def _build_event_mask(result:BatchEncoding, num_insts:int, event_start_token_id, event_end_token_id):

    event_tokens = []
    for i in range(num_insts):
        input_ids = result['input_ids'][i]
        try:
            event_start = input_ids.index(event_start_token_id)
        except:
            event_start = -1

        try:
            event_end = input_ids.index(event_end_token_id)
        except:
            event_end = len(input_ids)-1

        if event_start >= 0:
            inst_event_tokens = [0] * event_start + [1] * (event_end-event_start+1) + [0] * (len(input_ids)-event_end-1)
        else:
            inst_event_tokens = [1] * len(input_ids)

        event_tokens.append(inst_event_tokens)

    return event_tokens

def truncate_features(feature: Union[InputFeatures, HierarchicalInputFeatures]) -> str:
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
        "help": "A space-separated list of tasks to train on (mainly used as keys to internally track and display output)"})
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

    max_train_items: Optional[int] = field(
        default=-1, metadata={"help": "Set a number of train instances to use during training (useful for debugging data processing logic if a dataset is very large. Default is to train on all training data."}
    )

    max_eval_items: Optional[int] = field(
        default=-1, metadata={"help": "Set a number of validation instances to use during training (useful if a dataset has been created using dumb logic like 80/10/10 and 10%% takes forever to evaluate on. Default is evaluate on all validation data."}
    )


class ClinicalNlpDataset(Dataset):
    """
    Copy-pasted from GlueDataset with glue task-specific code changed;
    moved into here to be self-contained.

    :param args: the data training args for this experiment
    :param tokenizer: the tokenizer
    :param limit_length: if provided, the number of
        examples to include in the dataset
    :param cache_dir: if provided, the directory to save/load a cache
        of this dataset
    :param hierarchical: whether to structure the data for the hierarchical
        model (:class:`cnlpt.HierarchicalTransformer.HierarchicalModel`)
    """
    args: DataTrainingArguments
    features: List[InputFeatures]

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        hierarchical: bool = False,
    ):
        self.args = args
        self.processors = []
        self.class_weights = None
        self.tasks_to_labels = {}
        self.hierarchical = hierarchical
        self.datasets = []
        self.processed_dataset = []
        self.output_modes = {}

        # Load data features from cache or dataset file
        self.label_lists = []
        self.num_train_instances = 0

        if self.hierarchical:
            implicit_max_len = self.args.chunk_len * self.args.num_chunks
            if self.args.max_seq_length < implicit_max_len:
                raise ValueError('For the hierarchical model, the max seq length should be equal to the chunk length * num_chunks, otherwise what is the point?')

        # if cli supplies no tasks, the processor will assume we want all the tasks, but we do need to have a conventional order
        # for the model to use, so we need to still create a tasks variable.
        tasks = None if args.task_name is None else list(args.task_name)
        for data_dir_ind, data_dir in enumerate(args.data_dir):
            dataset_processor = AutoProcessor(data_dir, tasks, max_train_items=args.max_train_items)
            self.processors.append(dataset_processor)
            
            # Make sure that any overlapping task names have the same label sets and output mode definitions
            self._reconcile_labels_lists(dataset_processor)
            self._reconcile_output_modes(dataset_processor)

            self.datasets.append(dataset_processor.dataset)

        ## Each processed dataset will have been pruned to remove columns that the user _didn't_ ask for,
        ## but we need to add columns for tasks that it doesn't have that the user did ask for, with the 
        ## appropriate task types and  special values so pytorch knows not to compute losses over those 
        ## tasks for those inputs. Need to  do that after we've read all datasets and inferred all the task 
        ## types and label spaces.
        self._reconcile_columns()
        combined_dataset = self._concatenate_datasets()

        if tasks is None:
            # i.e., cli did not supply any task ordering:
            if 'train' in combined_dataset:
                split = 'train'
            elif 'dev' in combined_dataset:
                split = 'dev'
            else:
                split = 'test'

            tasks = list(combined_dataset[split].features.keys() - {'text', 'text_a', 'text_b'})

        self.tasks = tasks
        
        self.processed_dataset = combined_dataset.map(
            cnlp_preprocess_data,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset, organizing labels, creating hierarchical segments if necessary",
            batch_size=100,
            fn_kwargs = {
                'tokenizer':tokenizer,
                'max_length':args.max_seq_length,
                'label_lists':self.tasks_to_labels,
                'output_modes':self.output_modes,
                'inference': not 'train' in combined_dataset,
                'hierarchical':self.hierarchical,
                'chunk_len':self.args.chunk_len,
                'num_chunks':self.args.num_chunks,
                'insert_empty_chunk_at_beginning':self.args.insert_empty_chunk_at_beginning,
                'truncate_examples':self.args.truncate_examples,
                'tasks': tasks,
            }
        )

        if args.max_eval_items > 0:
            new_validation = self.processed_dataset['validation'].train_test_split(test_size=args.max_eval_items)['test']
            self.processed_dataset['validation'] = new_validation

        self.num_train_instances += self.processed_dataset['train'].num_rows

    def _reconcile_labels_lists(self, processor):
        '''
        given a new data processor, which extracted a label list for every task it contained,
        we reconcile it with existing label list for the same task
        '''
        for task,labels in processor.get_labels().items():
            if task in self.tasks_to_labels:
                ## a subset of existing labels (WARN), superset of existing labels (WARN), mixture, or no overlap (ERROR)
                new_labels = set(labels)
                old_labels = set(self.tasks_to_labels[task])
                if new_labels.isdisjoint(old_labels):
                    raise Exception("The same task name has disjoint sets of labels in different dataset: %s vs. %s" % (str(old_labels), str(new_labels)))
                elif new_labels != old_labels:
                    logger.warn("Two different datasets have the same task name but not completely equal label lists: %s vs. %s. We will merge them." % (str(old_labels), str(new_labels)))
                    self.tasks_to_labels[task] = old_labels.union(new_labels)
                else:
                    ## they match completely, nothing to be done
                    pass
            else:
                self.tasks_to_labels[task] = labels
    
    def _reconcile_output_modes(self, processor):
        '''
        given a new data processor, which inferred output modes for its tasks, make
        sure those output modes agree with existing inferred output modes for any
        same-named tasks.
        '''
        for task,output_mode in processor.get_output_modes().items():
            if task in self.output_modes:
                # There is an existing task with this name
                existing_output_mode = self.output_modes[task]
                if output_mode != existing_output_mode:
                    raise Exception("The task %s has two different output modes in different datasets and might not be the same task: %s vs. %s" % (task, existing_output_mode, output_mode))
            else:
                self.output_modes[task] = output_mode

    def _reconcile_columns(self):
        '''
        The overall dataset should get the superset of all task columns, even if some of the tasks aren't in
        all of the component datasets
        '''
        tasks = self.tasks_to_labels.keys()

        for dataset in self.datasets:
            for split_name in dataset.keys():
                for task in tasks:
                    if task not in dataset[split_name].column_names:
                        if self.output_modes[task] == tagging:
                            dataset[split_name] = dataset[split_name].add_column(task, [none_column] * len(dataset[split_name]))
                        elif self.output_modes[task] == relex:
                            pass
                        elif self.output_modes[task] == classification:
                            dataset[split_name] = dataset[split_name].add_column(task, [none_column] * len(dataset[split_name]))


                for column in dataset[split_name].column_names:
                    if column not in tasks and column not in text_columns:
                        dataset[split_name] = dataset[split_name].remove_columns(column)

    def _concatenate_datasets(self):
        """
        We have multiple dataset dicts, we need to create a single dataset dict
        where we concatenate each of the splits first.
        """
        datasets_by_split = {}
        for dataset in self.datasets:
            for split in dataset:
                if split not in datasets_by_split:
                    datasets_by_split[split] = []
                datasets_by_split[split].append(dataset[split])
        
        for split_name, split_data in datasets_by_split.items():
            datasets_by_split[split_name] = datasets.concatenate_datasets(split_data)
        
        return datasets.DatasetDict(datasets_by_split)

    def __len__(self) -> int:
        """
        Length method for this class.

        :return: the number of datasets included in this dataset
        """
        return len(self.datasets)

    def __getitem__(self, i) -> Union[InputFeatures, HierarchicalInputFeatures]:
        """
        Getitem method for this class.

        :param i: the index of the example to retrieve
        :return: the example at index `i`
        """
        return self.features[i]

    def get_labels(self) -> Dict[str, List[str]]:
        """
        Retrieve the label lists for all the tasks for the dataset.

        :return: the dictionary of label lists indexed by task name
        """
        return self.tasks_to_labels

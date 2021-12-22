import os
from os.path import basename, dirname
import time
import logging

from filelock import FileLock
from typing import Callable, Dict, Optional, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass, field
from enum import Enum

from .cnlp_processors import cnlp_processors, cnlp_output_modes, classification, tagging, relex, mtl

special_tokens = ['<e>', '</e>', '<a1>', '</a1>', '<a2>', '</a2>', '<cr>', '<neg>']

logger = logging.getLogger(__name__)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

class Split(Enum):
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
        return json.dumps(dataclasses.asdict(self)) + "\n"
    
def cnlp_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    token_classify=False,
):
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
            return list(label_map.values())[0]
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

    roberta_based = tokenizer.cls_token == '<s>'
    if not roberta_based:
        assert tokenizer.cls_token == '[CLS]', 'This tokenizer does not seem to be based on BERT or Roberta -- this will cause errors with the dataset encoding.'

    # This code has to solve the problem of properly setting labels for word pieces that do not actually need to be tagged.
    encoded_labels = []
    if output_mode == tagging:
        for sent_ind,sent in enumerate(sentences):
            sent_labels = []

            ## FIXME -- this is stupid and won't work outside the roberta encoding
            label_ind = 0
            for wp_ind,wp in enumerate(batch_encoding[sent_ind].tokens):
                if ((roberta_based and (wp.startswith('Ġ') or wp in special_tokens)) or 
                    (not roberta_based and not wp.startswith('[') and (not wp.startswith('##') or wp in special_tokens))):
                        sent_labels.append(labels[sent_ind].pop(0))
                else:
                    sent_labels.append(-100)
                # if wp_ind in word_inds:
                #     sent_labels.append(labels[sent_ind][label_ind])
                #     label_ind += 1
                # else:
                #     sent_labels.append(-100)
            
            encoded_labels.append(np.array(sent_labels))
   
        labels = encoded_labels
    elif output_mode == relex:
        # start by building a matrix that's N' x N' (word-piece length) with "None" as the default
        # for word pairs, and -100 (mask) as the default if one of word pair is a suffix token
        out_of_bounds = 0
        num_relations = 0
        for sent_ind, sent in enumerate(sentences):
            num_relations += len(labels[sent_ind])
            wpi_to_tokeni = {}
            tokeni_to_wpi = {}
            sent_labels = np.zeros( (max_length, max_length)) - 100
            wps = batch_encoding[sent_ind].tokens
            sent_len = len(wps)
            ## FIXME -- this is stupid and won't work outside the roberta encoding
            for wp_ind,wp in enumerate(wps):
                if wp.startswith('Ġ') or wp in special_tokens:
                    key = wp_ind
                    val = len(wpi_to_tokeni)

                    wpi_to_tokeni[key] = val
                    tokeni_to_wpi[val] = key
            
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
           
        feature = InputFeatures(**inputs, label=[labels[i]])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: List[str] = field(
        metadata={"help": "The input data dirs. A space-separated list of directories that should contain the .tsv files (or other data files) for the task. Should be presented in the same order as the task names."}
    )

    task_name: List[str] = field(default_factory=lambda: None, metadata={"help": "A space-separated list of tasks to train on: " + ", ".join(cnlp_processors.keys())})
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
        default=False, metadata={"help": "A flag that indicates whether class-specific loss should be used. This can be useful in cases with severe class imbalance. The formula for a weight of a class is the count of that class divided the count of the rarest class."}
    )


class ClinicalNlpDataset(Dataset):
    """ Copy-pasted from GlueDataset with glue task-specific code changed
        moved into here to be self-contained
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
    ):
        self.args = args
        self.processors = []
        self.output_mode = []
        self.class_weights = []

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

            cached_features_file = os.path.join(
                cache_dir if cache_dir is not None else data_dir,
                "cached_{}-{}_{}_{}_{}".format(
                    dataconfig, domain, mode.value, tokenizer.__class__.__name__, str(args.max_seq_length),
                ),
            )

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
                    )
                    start = time.time()
                    torch.save(features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    logger.info(
                        "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

                if self.args.weight_classes and mode == Split.train:
                    class_counts = [0] * len(self.label_lists[task_ind])
                    for feature in features:
                        labels = feature.label[0]
                        vals, counts = np.unique(labels, return_counts=True)
                        for val_ind,val in enumerate(vals):
                            if val >= 0:
                                class_counts[int(val)] += counts[val_ind]

                    self.class_weights[task_ind] = min(class_counts) / class_counts

                   
                if self.features is None:
                    self.features = features
                else:
                    # we should have all non-label features be the same, so we can essentially discard subsequent
                    # datasets input features. So we'll append the labels from that features list and discard the duplicate input features.
                    assert len(features) == len(self.features)
                    for feature_ind,feature in enumerate(features):
                        if len(self.features[feature_ind].label[0].shape) == 1 and len(feature.label[0].shape) == 1:
                            self.features[feature_ind].label[0] = np.stack([self.features[feature_ind].label[0],
                                                                        feature.label[0]])
                        else:
                            self.features[feature_ind].label[0] = np.concatenate([self.features[feature_ind].label[0],
                                                                        feature.label[0]])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_lists


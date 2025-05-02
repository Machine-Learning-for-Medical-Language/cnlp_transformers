# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import logging
import os
from contextlib import asynccontextmanager
from time import time
from typing import Union

import numpy as np
from fastapi import FastAPI
from nltk.tokenize import wordpunct_tokenize as tokenize
from pydantic import BaseModel
from seqeval.metrics.sequence_labeling import get_entities
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

from .utils import create_dataset, initialize_cnlpt_model

MODEL_NAME = "mlml-chip/thyme2_colon_e2e"
logger = logging.getLogger("Temporal_REST_Processor")
logger.setLevel(logging.INFO)

LABELS = ["-1", "1"]
TIMEX_LABEL_LIST = [
    "O",
    "B-DATE",
    "B-DURATION",
    "B-PREPOSTEXP",
    "B-QUANTIFIER",
    "B-SET",
    "B-TIME",
    "B-SECTIONTIME",
    "B-DOCTIME",
    "I-DATE",
    "I-DURATION",
    "I-PREPOSTEXP",
    "I-QUANTIFIER",
    "I-SET",
    "I-TIME",
    "I-SECTIONTIME",
    "I-DOCTIME",
]
TIMEX_LABEL_DICT = {val: ind for ind, val in enumerate(TIMEX_LABEL_LIST)}
EVENT_LABEL_LIST = [
    "O",
    "B-AFTER",
    "B-BEFORE",
    "B-BEFORE/OVERLAP",
    "B-OVERLAP",
    "I-AFTER",
    "I-BEFORE",
    "I-BEFORE/OVERLAP",
    "I-OVERLAP",
]
EVENT_LABEL_DICT = {val: ind for ind, val in enumerate(EVENT_LABEL_LIST)}

RELATION_LABEL_LIST = ["None", "CONTAINS", "OVERLAP", "BEFORE", "BEGINS-ON", "ENDS-ON"]
RELATION_LABEL_DICT = {val: ind for ind, val in enumerate(RELATION_LABEL_LIST)}

DTR_LABEL_LIST = ["AFTER", "BEFORE", "BEFORE/OVERLAP", "OVERLAP"]
OLD_DTR_LABEL_LIST = ["BEFORE", "OVERLAP", "BEFORE/OVERLAP", "AFTER"]

LABELS = [TIMEX_LABEL_LIST, EVENT_LABEL_LIST, RELATION_LABEL_LIST]
MAX_LENGTH = 128


class SentenceDocument(BaseModel):
    sentence: str


class TokenizedSentenceDocument(BaseModel):
    """sent_tokens: a list of sentences, where each sentence is a list of tokens"""

    sent_tokens: list[list[str]]
    metadata: str


class Timex(BaseModel):
    begin: int
    end: int
    timeClass: str


class Event(BaseModel):
    begin: int
    end: int
    dtr: str


class Relation(BaseModel):
    # Allow args to be none, so that we can potentially link them to times or events in the client, or if they don't
    # care about that. pass back the token indices of the args in addition.
    arg1: Union[str, None]
    arg2: Union[str, None]
    category: str
    arg1_start: int
    arg2_start: int


class TemporalResults(BaseModel):
    """lists of timexes, events and relations for list of sentences"""

    timexes: list[list[Timex]]
    events: list[list[Event]]
    relations: list[list[Relation]]


def create_instance_string(tokens: list[str]):
    return " ".join(tokens)


task_order: dict[str, int]
tasks: list[str]
tokenizer: PreTrainedTokenizer
trainer: Trainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        TIMEX_LABEL_LIST, \
        TIMEX_LABEL_DICT, \
        EVENT_LABEL_LIST, \
        EVENT_LABEL_DICT, \
        RELATION_LABEL_LIST, \
        RELATION_LABEL_DICT, \
        task_order, \
        tasks, \
        tokenizer, \
        trainer

    local_model_name = os.getenv("MODEL_NAME", MODEL_NAME)
    tokenizer, trainer = initialize_cnlpt_model(local_model_name)

    config_dict = trainer.model.config.to_dict()
    # For newer models (version >= 0.6.0), the label dictionary is saved with the model
    # config. we can look for it to preserve backwards compatibility for now but
    # should eventually remove the hardcoded label lists from our inference tools.
    label_dict = config_dict.get("label_dictionary", None)
    if label_dict is not None:
        # some older versions have one label dictionary per dataset, future versions should just
        # have a task-keyed dictionary
        if type(label_dict) is list:
            label_dict = label_dict[0]

        if "event" in label_dict:
            EVENT_LABEL_LIST = label_dict["event"]
            EVENT_LABEL_DICT = {val: ind for ind, val in enumerate(EVENT_LABEL_LIST)}
            print(EVENT_LABEL_LIST)

        if "timex" in label_dict:
            TIMEX_LABEL_LIST = label_dict["timex"]
            TIMEX_LABEL_DICT = {val: ind for ind, val in enumerate(TIMEX_LABEL_LIST)}
            print(TIMEX_LABEL_LIST)

        if "tlinkx" in label_dict:
            RELATION_LABEL_LIST = label_dict["tlinkx"]
            RELATION_LABEL_DICT = {
                val: ind for ind, val in enumerate(RELATION_LABEL_LIST)
            }
            print(RELATION_LABEL_LIST)

    tasks = config_dict.get("finetuning_task", None)
    task_order = {}
    if tasks is not None:
        print("Overwriting finetuning task order")
        for task_ind, task_name in enumerate(tasks):
            task_order[task_name] = task_ind
        print(task_order)
    else:
        print("Didn't find a new task ordering in the model config")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/temporal/process")
async def process(doc: TokenizedSentenceDocument):
    return process_tokenized_sentence_document(doc)


@app.post("/temporal/process_sentence")
async def process_sentence(doc: SentenceDocument):
    tokenized_sent = tokenize(doc.sentence)
    doc = TokenizedSentenceDocument(
        sent_tokens=[
            tokenized_sent,
        ],
        metadata="Single sentence",
    )
    return process_tokenized_sentence_document(doc)


def process_tokenized_sentence_document(doc: TokenizedSentenceDocument):
    sents = doc.sent_tokens
    metadata = doc.metadata

    print(EVENT_LABEL_LIST)
    print(TIMEX_LABEL_LIST)
    print(RELATION_LABEL_LIST)

    logger.warning(f"Received document labeled {metadata} with {len(sents)} sentences")
    instances = []
    start_time = time()

    for sent_ind, token_list in enumerate(sents):
        inst_str = create_instance_string(token_list)
        logger.debug(f"Instance string is {inst_str}")
        instances.append(inst_str)

    dataset = create_dataset(instances, tokenizer, MAX_LENGTH)
    preproc_end = time()

    output = trainer.predict(test_dataset=dataset)

    timex_predictions = np.argmax(output.predictions[task_order["timex"]], axis=2)
    event_predictions = np.argmax(output.predictions[task_order["event"]], axis=2)
    rel_predictions = np.argmax(output.predictions[task_order["tlinkx"]], axis=3)
    rel_inds = np.where(rel_predictions != RELATION_LABEL_DICT["None"])

    logging.debug(f"Found relation indices: {rel_inds!s}")

    rels_by_sent = {}
    for rel_num in range(len(rel_inds[0])):
        sent_ind = rel_inds[0][rel_num]
        if sent_ind not in rels_by_sent:
            rels_by_sent[sent_ind] = []

        arg1_ind = rel_inds[1][rel_num]
        arg2_ind = rel_inds[2][rel_num]
        if arg1_ind == arg2_ind:
            # no relations between an entity and itself
            logger.warning("Found relation between an entity and itself... skipping")
            continue

        rel_cat = rel_predictions[sent_ind, arg1_ind, arg2_ind]

        rels_by_sent[sent_ind].append((arg1_ind, arg2_ind, rel_cat))

    pred_end = time()

    timex_results = []
    event_results = []
    rel_results = []

    for sent_ind in range(len(dataset)):
        batch_encoding = tokenizer(
            [
                sents[sent_ind],
            ],
            is_split_into_words=True,
            max_length=MAX_LENGTH,
        )
        word_ids = batch_encoding.word_ids(0)
        wpind_to_ind = {}
        timex_labels = []
        event_labels = []
        previous_word_idx = None

        for word_pos_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and word_idx is not None:
                key = word_pos_idx
                val = len(wpind_to_ind)

                wpind_to_ind[key] = val
                # tokeni_to_wpi[val] = key
                timex_labels.append(
                    TIMEX_LABEL_LIST[timex_predictions[sent_ind][word_pos_idx]]
                )
                try:
                    event_labels.append(
                        EVENT_LABEL_LIST[event_predictions[sent_ind][word_pos_idx]]
                    )
                except Exception as e:
                    print(
                        f"exception thrown when sent_ind={sent_ind} and word_pos_idx={word_pos_idx}"
                    )
                    print(
                        f"prediction is {event_predictions[sent_ind][word_pos_idx]!s}"
                    )
                    raise e

            previous_word_idx = word_idx

        timex_entities = get_entities(timex_labels)
        logging.info(
            f"Extracted {len(timex_entities)} timex entities from the sentence"
        )
        timex_results.append(
            [
                Timex(timeClass=label[0], begin=label[1], end=label[2])
                for label in timex_entities
            ]
        )

        event_entities = get_entities(event_labels)
        logging.info(f"Extracted {len(event_entities)} events from the sentence")
        event_results.append(
            [
                Event(dtr=label[0], begin=label[1], end=label[2])
                for label in event_entities
            ]
        )

        rel_sent_results = []
        for rel in rels_by_sent.get(sent_ind, []):
            arg1 = None
            arg2 = None
            if rel[0] not in wpind_to_ind or rel[1] not in wpind_to_ind:
                logging.warning(
                    "Found a relation to a non-leading wordpiece token... ignoring"
                )
                continue

            arg1_ind = wpind_to_ind[rel[0]]
            arg2_ind = wpind_to_ind[rel[1]]

            sent_timexes = timex_results[-1]
            for timex_ind, timex in enumerate(sent_timexes):
                if timex.begin == arg1_ind:
                    arg1 = f"TIMEX-{timex_ind}"
                if timex.begin == arg2_ind:
                    arg2 = f"TIMEX-{timex_ind}"

            sent_events = event_results[-1]
            for event_ind, event in enumerate(sent_events):
                if event.begin == arg1_ind:
                    arg1 = f"EVENT-{event_ind}"
                if event.begin == arg2_ind:
                    arg2 = f"EVENT-{event_ind}"

            rel = Relation(
                arg1=arg1,
                arg2=arg2,
                category=RELATION_LABEL_LIST[rel[2]],
                arg1_start=arg1_ind,
                arg2_start=arg2_ind,
            )
            rel_sent_results.append(rel)

        rel_results.append(rel_sent_results)

    results = TemporalResults(
        timexes=timex_results, events=event_results, relations=rel_results
    )

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info(
        f"Pre-processing time: {preproc_time:f}, processing time: {pred_time:f}, post-processing time {postproc_time:f}"
    )

    return results

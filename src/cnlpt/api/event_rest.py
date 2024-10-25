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
from contextlib import asynccontextmanager
from time import time

import numpy as np
from fastapi import FastAPI
from nltk.tokenize import wordpunct_tokenize as tokenize
from seqeval.metrics.sequence_labeling import get_entities
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

from .cnlp_rest import create_dataset, initialize_cnlpt_model
from .temporal_rest import (
    EVENT_LABEL_LIST,
    Event,
    SentenceDocument,
    TemporalResults,
    TokenizedSentenceDocument,
    create_instance_string,
)

MODEL_NAME = "tmills/event-thyme-colon-pubmedbert"
logger = logging.getLogger("Event_REST_Processor")
logger.setLevel(logging.INFO)

MAX_LENGTH = 128


tokenizer: PreTrainedTokenizer
trainer: Trainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, trainer
    tokenizer, trainer = initialize_cnlpt_model(MODEL_NAME)
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

    logger.warning(f"Received document labeled {metadata} with {len(sents)} sentences")
    instances = []
    start_time = time()

    for sent_ind, token_list in enumerate(sents):
        inst_str = create_instance_string(token_list)
        logger.debug(f"Instance string is {inst_str}")
        instances.append(inst_str)

    dataset = create_dataset(instances, tokenizer, max_length=MAX_LENGTH)
    preproc_end = time()

    output = trainer.predict(test_dataset=dataset)

    event_predictions = np.argmax(output.predictions[0], axis=2)

    pred_end = time()

    timex_results = []
    event_results = []
    rel_results = []

    for sent_ind in range(len(dataset)):
        batch_encoding = tokenizer.batch_encode_plus(
            [
                sents[sent_ind],
            ],
            is_split_into_words=True,
            max_length=MAX_LENGTH,
        )
        word_ids = batch_encoding.word_ids(0)
        wpind_to_ind = {}
        event_labels = []
        previous_word_idx = None

        for word_pos_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and word_idx is not None:
                key = word_pos_idx
                val = len(wpind_to_ind)

                wpind_to_ind[key] = val
                event_labels.append(
                    EVENT_LABEL_LIST[event_predictions[sent_ind][word_pos_idx]]
                )
            previous_word_idx = word_idx

        event_entities = get_entities(event_labels)
        logging.info(f"Extracted {len(event_entities)} events from the sentence")
        event_results.append(
            [
                Event(dtr=label[0], begin=label[1], end=label[2])
                for label in event_entities
            ]
        )
        timex_results.append([])
        rel_results.append([])

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


@app.post("/temporal/collection_process_complete")
async def collection_process_complete():
    global trainer
    trainer = None


def rest():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the http server for temporal event extraction"
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="The port number to run the server on",
        default=8000,
    )

    args = parser.parse_args()

    import uvicorn

    uvicorn.run("cnlpt.api.event_rest:app", host="0.0.0.0", port=args.port, reload=True)


if __name__ == "__main__":
    rest()

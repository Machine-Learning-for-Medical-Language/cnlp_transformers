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

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict

from transformers.data.processors.utils import InputFeatures, InputExample
from torch.utils.data.dataset import Dataset
import numpy as np
from .cnlp_rest import initialize_cnlpt_model, get_dataset
from ..CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from seqeval.metrics.sequence_labeling import get_entities
import logging
from time import time
from nltk.tokenize import wordpunct_tokenize as tokenize

from .temporal_rest import timex_label_list, TokenizedSentenceDocument, SentenceDocument, Timex, TemporalResults,  create_instance_string

app = FastAPI()
model_name = "tmills/timex-thyme-colon-pubmedbert"
logger = logging.getLogger('Timex_REST_Processor')
logger.setLevel(logging.INFO)

max_length = 128

@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, model_name)

@app.post("/temporal/process")
async def process(doc: TokenizedSentenceDocument):
    return process_tokenized_sentence_document(doc)

@app.post("/temporal/process_sentence")
async def process_sentence(doc: SentenceDocument):
    tokenized_sent = tokenize(doc.sentence)
    doc = TokenizedSentenceDocument(sent_tokens=[tokenized_sent,], metadata='Single sentence')
    return process_tokenized_sentence_document(doc)

def process_tokenized_sentence_document(doc: TokenizedSentenceDocument):
    sents = doc.sent_tokens
    metadata = doc.metadata

    logger.warn('Received document labeled %s with %d sentences' % (metadata, len(sents)))
    instances = []
    start_time = time()

    for sent_ind, token_list in enumerate(sents):
        inst_str = create_instance_string(token_list)
        logger.debug('Instance string is %s' % (inst_str))
        instances.append(inst_str)

    dataset = get_dataset(instances, app.state.tokenizer, label_lists=[timex_label_list], tasks=['timex'], max_length=max_length)
    logger.warn('Dataset is as follows: %s' % (str(dataset.features)))

    preproc_end = time()

    output = app.state.trainer.predict(test_dataset=dataset)

    timex_predictions = np.argmax(output.predictions[0], axis=2)
    
    timex_results = []
    event_results = []
    relation_results = []

    pred_end = time()
    
    for sent_ind in range(len(dataset)):
        batch_encoding = app.state.tokenizer.batch_encode_plus([sents[sent_ind],],
                                                           is_split_into_words=True,
                                                           max_length=max_length)
        word_ids = batch_encoding.word_ids(0)
        wpind_to_ind = {}
        timex_labels = []
        previous_word_idx = None

        for word_pos_idx, word_idx in enumerate(word_ids):
            if word_idx != previous_word_idx and word_idx is not None:
                key = word_pos_idx
                val = len(wpind_to_ind)

                wpind_to_ind[key] = val
                timex_labels.append(timex_label_list[timex_predictions[sent_ind][word_pos_idx]])
            previous_word_idx = word_idx

        timex_entities = get_entities(timex_labels)
        logging.info("Extracted %d timex entities from the sentence" % (len(timex_entities)))
        timex_results.append( [Timex(timeClass=label[0], begin=label[1], end=label[2]) for label in timex_entities] )
        event_results.append( [] )
        relation_results.append( [] )


    results = TemporalResults(timexes=timex_results, events=event_results, relations=relation_results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))

    return results

def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for temporal')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("cnlpt.api.timex_rest:app", host='0.0.0.0', port=args.port, reload=True)


if __name__ == '__main__':
    rest()

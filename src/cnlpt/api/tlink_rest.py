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

from fastapi import FastAPI
import numpy as np
from cnlp_rest import create_instance_string, initialize_cnlpt_model, create_dataset
import logging
from time import time
from nltk.tokenize import wordpunct_tokenize as tokenize

from temporal_rest import TokenizedSentenceDocument, SentenceDocument, Relation, TemporalResults, create_instance_string

app = FastAPI()
model_name = "../../../../../thyme/ft/models/inference/tlink/5688819"
logger = logging.getLogger('Tlink_REST_Processor')
logger.setLevel(logging.INFO)

max_length = 128

relation_label_list = ["AFTER", "BEFORE", "BEGINS-ON", "BEGINS-ON-1", "CONTAINS", "CONTAINS-1", "ENDS-ON", "ENDS-ON-1", "NOTED-ON", "NOTED-ON-1", "None", "OVERLAP"]

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

    dataset = create_dataset(instances, app.state.tokenizer, label_lists=[relation_label_list], tasks=['tlink'], max_length=max_length)
    preproc_end = time()

    output = app.state.trainer.predict(test_dataset=dataset)

    rel_predictions = np.argmax(output.predictions[0], axis=1)

    pred_end = time()

    timex_results = []
    event_results = []
    rel_results = []

    for sent_ind in range(len(dataset)):
        event_results.append( [] )
        timex_results.append( [] )
        rel_results.append( [Relation(arg1=None, arg2=None, category=relation_label_list[rel_predictions[sent_ind]], arg1_start=sents[sent_ind].index("<e1>"), arg2_start=sents[sent_ind].index("<e2>"))] )


    results = TemporalResults(timexes=timex_results, events=event_results, relations=rel_results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))

    return results


@app.post("/temporal/collection_process_complete")
async def collection_process_complete():
    app.state.trainer = None

def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for temporal relation extraction')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("tlink_rest:app", host='0.0.0.0', port=args.port, reload=True)


if __name__ == '__main__':
    rest()


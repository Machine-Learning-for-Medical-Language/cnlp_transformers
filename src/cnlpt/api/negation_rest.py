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
from pydantic import BaseModel

from typing import List, Tuple, Dict

from .cnlp_rest import EntityDocument, create_instance_string, initialize_cnlpt_model, get_dataset
from ..CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
import numpy as np

import logging
from time import time

app = FastAPI()
model_name = "tmills/cnlpt-negation-roberta-sharpseed"
logger = logging.getLogger('Negation_REST_Processor')
logger.setLevel(logging.DEBUG)

task = 'Negation'
labels = [-1, 1]

max_length = 128

class NegationResults(BaseModel):
    ''' statuses: dictionary from entity id to classification decision about negation; true -> negated, false -> not negated'''
    statuses: List[int]

@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, model_name)

@app.post("/negation/process")
async def process(doc: EntityDocument):
    doc_text = doc.doc_text
    logger.warn('Received document of len %d to process with %d entities' % (len(doc_text), len(doc.entities)))
    instances = []
    start_time = time()

    for ent_ind, offsets in enumerate(doc.entities):
        # logger.debug('Entity ind: %d has offsets (%d, %d)' % (ent_ind, offsets[0], offsets[1]))
        inst_str = create_instance_string(doc_text, offsets)
        logger.debug('Instance string is %s' % (inst_str))
        instances.append(inst_str)

    dataset = get_dataset(instances, app.state.tokenizer, [labels,], [task,], max_length)
    preproc_end = time()


    output = app.state.trainer.predict(test_dataset=dataset)
    predictions = output.predictions[0]
    predictions = np.argmax(predictions, axis=1)

    pred_end = time()

    results = []
    for ent_ind in range(len(dataset)):
        results.append(labels[predictions[ent_ind]])

    output = NegationResults(statuses=results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.warn("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))
    
    return output

@app.get("/negation/{test_str}")
async def test(test_str: str):
    return {'argument': test_str}


def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for negation')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("cnlpt.api.negation_rest:app", host='0.0.0.0', port=args.port, reload=True)


if __name__ == '__main__':
    rest()

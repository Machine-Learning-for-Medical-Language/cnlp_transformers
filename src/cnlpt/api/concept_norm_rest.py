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

# Modeling imports
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

# from .api.cnlp_rest import get_dataset
from datasets import Dataset

from ..CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from .cnlp_rest import  get_dataset, initialize_cnlpt_model
import numpy as np
import torch

import logging, os, json
from time import time

app = FastAPI()
model_name = "/lab-share/CHIP-Savova-e2/Public/resources/cnlpt/concept_norm/share/checkpoint-57456/"
logger = logging.getLogger('Concept_Normalization_REST_Processor')
logger.setLevel(logging.DEBUG)

task = 'conceptnorm'
with open(os.path.join(model_name,"../ontology_cui.txt"), 'r') as outfile:
    labels = json.load(outfile)

outfile.close()
labels = labels + ["CUI-less"]

max_length = 32

class Entity(BaseModel):
    ''' doc_text: The raw text of the document
    offset:  A list of entities, where each is a tuple of character offsets into doc_text for that entity'''
    entity_text: str


class ConceptNormResults(BaseModel):
    ''' statuses: dictionary from entity id to classification decision about negation; true -> negated, false -> not negated'''
    statuses: List[str]

@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, model_name)

@app.post("/concept_norm/process")
async def process(entity: Entity):
    text = entity.entity_text
    logger.warn('Received entities of len %d to process' % (len(text)))
    instances = [text]
    start_time = time()

    dataset = get_dataset(instances, app.state.tokenizer, [labels,], [task,], max_length)
    preproc_end = time()

    output = app.state.trainer.predict(test_dataset=dataset)
    predictions = output.predictions[0]
    predictions = np.argmax(predictions, axis=1)

    pred_end = time()

    results = []
    for ent_ind in range(len(dataset)):
        results.append(labels[predictions[ent_ind]])

    output = ConceptNormResults(statuses=results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.warn("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))
    
    return output

@app.get("/conceptnorm/{test_str}")
async def test(test_str: str):
    return {'argument': test_str}


def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for negation')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("cnlpt.api.concept_norm_rest:app", host='0.0.0.0', port=args.port, reload=True)

if __name__ == '__main__':
    rest()
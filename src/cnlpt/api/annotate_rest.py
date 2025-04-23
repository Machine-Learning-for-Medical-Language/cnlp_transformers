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
from pydantic import BaseModel


from nltk.tokenize import wordpunct_tokenize as tokenize # from timex rest
from seqeval.metrics.sequence_labeling import get_entities # from timex rest

from .utils import (
    EntityDocument,
    create_dataset,
    create_instance_string,
    initialize_cnlpt_model,
)


logger = logging.getLogger("DoNotAnnotate_REST_Processor")
logger.setLevel(logging.DEBUG)


MODEL_PATH = "/temp"  
TASK = "DoNotAnnotate"
LABELS = [0, 1]  # Do not annotate = 0, Annotate = 1

MAX_LENGTH = 128

class AnnotateResults(BaseModel):
    """statuses: list of classifier outputs for every input"""
    labels: list[int]

tokenizer: PreTrainedTokenizer
trainer: Trainer

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, trainer
    tokenizer, trainer = initialize_cnlpt_model(MODEL_NAME)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/annotate/process")
async def process(doc: UnannotatedDocument):
    lines = [line.strip() for line in doc_text.split("\n") if line.strip()]
    logger.warning(
        f"Received document of {len(doc_text)} to process with {len(lines)} non-empty lines"
        )
    start_time = time()
    
    if not lines:
        return AnnotateResults(labels=[])

    dataset = create_dataset(lines, tokenizer, MAX_LENGTH) 
    preproc_end = time()

    output = trainer.predict(test_dataset=dataset)
    predictions = np.argmax(output.predictions, axis=1)
    
    pred_end = time()

    results = AnnotateResults(labels=predictions.tolist())

    results = [] 
    for ind in range(len(dataset)): 
        results.append(LABELS[predictions[ind]]) 
        
    output = AnnotateResults(statuses=results) 

    postproc_end = time()
    
    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info(
        f"Pre-processing time: {preproc_time:f}, processing time: {pred_time:f}, post-processing time {postproc_time:f}"
    )

    return output 
 

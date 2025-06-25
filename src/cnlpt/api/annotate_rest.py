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
import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

from cnlpt.api.utils import (
    UnannotatedDocument,
    get_dataset,
    initialize_cnlpt_model,
)


logger = logging.getLogger("DoNotAnnotate_REST_Processor")
logger.setLevel(logging.DEBUG)


MODEL_PATH = "annotate_model"
TASK = "DoNotAnnotate"
LABELS = [0, 1]  # Do not annotate = 0, Annotate = 1

MAX_LENGTH = 128

class AnnotateResults(BaseModel):
    """labels: list of classifier outputs for every input"""
    labels: list[int]

tokenizer: PreTrainedTokenizer
trainer: Trainer
#tokenizer, trainer = initialize_cnlpt_model(app, MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, trainer
    tokenizer, trainer = initialize_cnlpt_model(app, MODEL_PATH)
    yield

# app = FastAPI(lifespan=lifespan)
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, os.getenv("MODEL_PATH", MODEL_PATH))

def return_labels(doc: UnannotatedDocument):
    lines = doc.doc_text.splitlines(keepends=True)
    logger.warning(
        f"Received document of {len(doc.doc_text)} to process with {len(lines)} non-empty lines"
        )
    start_time = time()
    
    if not lines:
        return AnnotateResults(labels=[])

    dataset = get_dataset(lines, app.state.tokenizer, LABELS, [TASK], MAX_LENGTH) 
    preproc_end = time()

    output = app.state.trainer.predict(test_dataset=dataset)
    predictions = output.predictions[0]
    predictions = np.argmax(predictions, axis=1)
    
    pred_end = time()
    
    results = []
    for ent_ind in range(len(dataset)):
        results.append(LABELS[predictions[ent_ind]])
    
    output = AnnotateResults(labels=results).labels 

    postproc_end = time()
    
    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logger.info(
        f"Pre-processing time: {preproc_time:f}, processing time: {pred_time:f}, post-processing time {postproc_time:f}"
    )

    return output
 
@app.post("/annotate/process")
async def process(doc: UnannotatedDocument):
    labels = return_labels(doc)
    offsets_list = []
    current_start_line = None
    char_index = 0

    for label, line in zip(labels, doc.doc_text.splitlines(keepends=True)):
        line_len = len(line)
        if label == 0:
            if current_start_line is None:
                current_start_line = char_index
            current_end_line = char_index + line_len
        else:
            if current_start_line is not None:
                offsets_list.append((current_start_line, current_end_line))
                current_start_line = None
        char_index += line_len
    
    # Catch any trailing 0 label group
    if current_start_line is not None:
        offsets_list.append((current_start_line, current_end_line))

    
    return offsets_list


def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for extraction of do-not-annotate spans')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("annotate_rest:app", host='0.0.0.0', port=args.port, reload=True)


if __name__ == "__main__":
    rest()

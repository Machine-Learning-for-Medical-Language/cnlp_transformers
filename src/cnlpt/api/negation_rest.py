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
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizer

from .utils import (
    EntityDocument,
    create_dataset,
    create_instance_string,
    initialize_cnlpt_model,
)

MODEL_NAME = "mlml-chip/negation_pubmedbert_sharpseed"
logger = logging.getLogger("Negation_REST_Processor")
logger.setLevel(logging.DEBUG)

TASK = "Negation"
LABELS = [-1, 1]

MAX_LENGTH = 128


class NegationResults(BaseModel):
    """statuses: dictionary from entity id to classification decision about negation; true -> negated, false -> not negated"""

    statuses: list[int]


tokenizer: PreTrainedTokenizer
trainer: Trainer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, trainer
    tokenizer, trainer = initialize_cnlpt_model(MODEL_NAME)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/negation/process")
async def process(doc: EntityDocument):
    doc_text = doc.doc_text
    logger.warning(
        f"Received document of len {len(doc_text)} to process with {len(doc.entities)} entities"
    )
    instances = []
    start_time = time()

    if len(doc.entities) == 0:
        return NegationResults(statuses=[])

    for ent_ind, offsets in enumerate(doc.entities):
        # logger.debug('Entity ind: %d has offsets (%d, %d)' % (ent_ind, offsets[0], offsets[1]))
        inst_str = create_instance_string(doc_text, offsets)
        logger.debug(f"Instance string is {inst_str}")
        instances.append(inst_str)

    dataset = create_dataset(instances, tokenizer, MAX_LENGTH)
    preproc_end = time()

    output = trainer.predict(test_dataset=dataset)
    predictions = output.predictions[0]
    predictions = np.argmax(predictions, axis=1)

    pred_end = time()

    results = []
    for ent_ind in range(len(dataset)):
        results.append(LABELS[predictions[ent_ind]])

    output = NegationResults(statuses=results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.warning(
        f"Pre-processing time: {preproc_time:f}, processing time: {pred_time:f}, post-processing time {postproc_time:f}"
    )

    return output


@app.get("/negation/{test_str}")
async def test(test_str: str):
    return {"argument": test_str}

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
from time import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .cnlp_rest import (
    EntityDocument,
    create_instance_string,
    get_dataset,
    initialize_cnlpt_model,
)

app = FastAPI()
model_name = "mlml-chip/sharpseed-termexists"
logger = logging.getLogger("TermExists_REST_Processor")
logger.setLevel(logging.DEBUG)

task = "TermExists"
labels = [-1, 1]

max_length = 128


class TermExistsResults(BaseModel):
    """statuses: list of classifier outputs for every input"""

    statuses: list[int]


@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, model_name)


@app.post("/termexists/process")
async def process(doc: EntityDocument):
    doc_text = doc.doc_text
    logger.warning(
        f"Received document of len {len(doc_text)} to process with {len(doc.entities)} entities"
    )
    instances = []
    start_time = time()

    if len(doc.entities) == 0:
        return TermExistsResults(statuses=[])

    for ent_ind, offsets in enumerate(doc.entities):
        inst_str = create_instance_string(doc_text, offsets)
        logger.debug(f"Instance string is {inst_str}")
        instances.append(inst_str)

    dataset = get_dataset(instances, app.state.tokenizer, max_length)
    preproc_end = time()

    output = app.state.trainer.predict(test_dataset=dataset)
    predictions = output.predictions[0]
    predictions = np.argmax(predictions, axis=1)

    pred_end = time()

    results = []
    for ent_ind in range(len(dataset)):
        results.append(labels[predictions[ent_ind]])

    output = TermExistsResults(statuses=results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.warning(
        f"Pre-processing time: {preproc_time:f}, processing time: {pred_time:f}, post-processing time {postproc_time:f}"
    )

    return output


def rest():
    import argparse

    parser = argparse.ArgumentParser(description="Run the http server for term exists")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="The port number to run the server on",
        default=8000,
    )

    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "cnlpt.api.termexists_rest:app", host="0.0.0.0", port=args.port, reload=True
    )


if __name__ == "__main__":
    rest()

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
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from os.path import join
from typing import Any

import numpy as np
import torch
import torch.backends.mps
from fastapi import FastAPI
from scipy.special import softmax
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..BaselineModels import CnnSentenceClassifier
from .cnlp_rest import UnannotatedDocument, create_dataset, resolve_device

MODEL_NAME = os.getenv("MODEL_PATH")
if MODEL_NAME is None:
    sys.stderr.write("This REST container requires a MODEL_PATH environment variable\n")
    sys.exit(-1)
device = os.getenv("MODEL_DEVICE", "auto")
device = resolve_device(device)

logger = logging.getLogger("CNN_REST_Processor")
logger.setLevel(logging.DEBUG)

MAX_SEQ_LENGTH = 128

model: CnnSentenceClassifier
tokenizer: PreTrainedTokenizer
conf_dict: dict[str, Any]


@asynccontextmanager
async def lifespan():
    global model, tokenizer, conf_dict
    conf_file = join(MODEL_NAME, "config.json")
    with open(conf_file) as fp:
        conf_dict = json.load(fp)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_labels_dict = {
        task: len(values) for task, values in conf_dict["label_dictionary"].items()
    }
    model = CnnSentenceClassifier.from_pretrained(
        MODEL_NAME,
        vocab_size=len(tokenizer),
        task_names=conf_dict["task_names"],
        num_labels_dict=num_labels_dict,
    )

    model = model.to(device)
    tokenizer = tokenizer
    conf_dict = conf_dict

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/cnn/classify")
async def process(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = create_dataset(
        instances, app.state.tokenizer, max_length=app.state.conf_dict["max_seq_length"]
    )
    _, logits = app.state.model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to(device),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to(device),
    )

    prediction = int(np.argmax(logits[0].cpu().detach().numpy(), axis=1))
    result = conf_dict["label_dictionary"][conf_dict["task_names"][0]][prediction]
    probabilities = softmax(logits[0][0].cpu().detach().numpy())
    # for redcap purposes, it might make more sense to only output the probability for the predicted class,
    # but i'm outputting them all, for transparency
    out_probabilities = [str(prob) for prob in probabilities]
    return {"result": result, "probabilities": out_probabilities}


def rest():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the http server for serving CNN model outputs."
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

    uvicorn.run("cnlpt.api.cnn_rest:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    rest()

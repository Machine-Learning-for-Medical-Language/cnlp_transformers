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
from os.path import join

import numpy as np
import torch
from fastapi import FastAPI
from scipy.special import softmax
from transformers import AutoTokenizer, Trainer

from ..BaselineModels import CnnSentenceClassifier
from .cnlp_rest import UnannotatedDocument, get_dataset

app = FastAPI()
model_name = os.getenv("MODEL_PATH")
if model_name is None:
    sys.stderr.write("This REST container requires a MODEL_PATH environment variable\n")
    sys.exit(-1)

logger = logging.getLogger("CNN_REST_Processor")
logger.setLevel(logging.DEBUG)

max_seq_length = 128


@app.on_event("startup")
async def startup_event():
    conf_file = join(model_name, "config.json")
    with open(conf_file, "rt") as fp:
        conf_dict = json.load(fp)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels_dict = {
        task: len(values) for task, values in conf_dict["label_dictionary"].items()
    }
    model = CnnSentenceClassifier.from_pretrained(
        model_name,
        vocab_size=len(tokenizer),
        task_names=conf_dict["task_names"],
        num_labels_dict=num_labels_dict,
    )

    app.state.model = model.to("cuda")
    app.state.tokenizer = tokenizer
    app.state.conf_dict = conf_dict


@app.post("/cnn/classify")
async def process(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = get_dataset(instances, app.state.tokenizer, max_length=max_seq_length)
    _, logits = app.state.model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to("cuda"),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to("cuda"),
    )
    prediction = int(np.argmax(logits[0].cpu().detach().numpy(), axis=1))
    result = app.state.conf_dict["label_dictionary"][
        app.state.conf_dict["task_names"][0]
    ][prediction]
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

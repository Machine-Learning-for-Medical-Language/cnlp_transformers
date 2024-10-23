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
import os

import torch
from fastapi import FastAPI

from .cnlp_rest import (
    UnannotatedDocument,
    get_dataset,
    initialize_hier_model,
)

app = FastAPI()
model_name = os.getenv("MODEL_PATH")

logger = logging.getLogger("HierRep_REST_Processor")
logger.setLevel(logging.DEBUG)


@app.on_event("startup")
async def startup_event():
    initialize_hier_model(app, model_name=model_name)


@app.post("/hier/get_rep")
async def get_representation(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = get_dataset(
        instances,
        app.state.tokenizer,
        max_length=16000,
        hier=True,
        chunk_len=200,
        num_chunks=80,
        insert_empty_chunk_at_beginning=False,
    )
    result = app.state.model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to("cuda"),
        token_type_ids=torch.LongTensor(dataset["token_type_ids"]).to("cuda"),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to("cuda"),
        output_hidden_states=True,
    )

    # Convert to a list so python can send it out
    hidden_states = result["hidden_states"].to("cpu").detach().numpy()[:, 0, :].tolist()
    return {"reps": hidden_states[0]}


@app.post("/hier/classify")
async def classify(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = get_dataset(
        instances,
        app.state.tokenizer,
        max_length=16000,
        hier=True,
        chunk_len=200,
        num_chunks=80,
        insert_empty_chunk_at_beginning=False,
    )
    result = app.state.model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to("cuda"),
        token_type_ids=torch.LongTensor(dataset["token_type_ids"]).to("cuda"),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to("cuda"),
        output_hidden_states=False,
    )

    predictions = [
        int(torch.argmax(logits.to("cpu").detach()).numpy())
        for logits in result["logits"]
    ]
    labels = [
        list(app.state.model.label_dictionary.values())[0][x] for x in predictions
    ]
    return {"result": labels}


def rest():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the http server for serving hierarchical model outputs."
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

    uvicorn.run("cnlpt.api.hier_rest:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    rest()

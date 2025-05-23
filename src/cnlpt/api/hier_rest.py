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
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from .utils import (
    UnannotatedDocument,
    create_dataset,
    initialize_hier_model,
    resolve_device,
)

MODEL_NAME = os.getenv("MODEL_PATH")

device = os.getenv("MODEL_DEVICE", "auto")
device = resolve_device(device)

logger = logging.getLogger("HierRep_REST_Processor")
logger.setLevel(logging.DEBUG)

tokenizer: PreTrainedTokenizer
model: PreTrainedModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    tokenizer, model = initialize_hier_model(MODEL_NAME)
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/hier/get_rep")
async def get_representation(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = create_dataset(
        instances,
        tokenizer,
        max_length=16000,
        hier=True,
        chunk_len=200,
        num_chunks=80,
        insert_empty_chunk_at_beginning=False,
    )

    result = model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to(model.device),
        token_type_ids=torch.LongTensor(dataset["token_type_ids"]).to(model.device),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to(model.device),
        output_hidden_states=True,
    )

    # Convert to a list so python can send it out
    hidden_states = result["hidden_states"].to("cpu").detach().numpy()[:, 0, :].tolist()
    return {"reps": hidden_states[0]}


@app.post("/hier/classify")
async def classify(doc: UnannotatedDocument):
    instances = [doc.doc_text]
    dataset = create_dataset(
        instances,
        tokenizer,
        max_length=16000,
        hier=True,
        chunk_len=200,
        num_chunks=80,
        insert_empty_chunk_at_beginning=False,
    )
    result = model.forward(
        input_ids=torch.LongTensor(dataset["input_ids"]).to(model.device),
        token_type_ids=torch.LongTensor(dataset["token_type_ids"]).to(model.device),
        attention_mask=torch.LongTensor(dataset["attention_mask"]).to(model.device),
        output_hidden_states=False,
    )

    predictions = [
        int(torch.argmax(logits.to("cpu").detach()).numpy())
        for logits in result["logits"]
    ]
    labels = [next(iter(model.label_dictionary.values()))[x] for x in predictions]
    return {"result": labels}

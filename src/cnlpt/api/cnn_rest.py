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

import logging
import os
from os.path import join
import sys
import json

from transformers import AutoTokenizer, Trainer
import torch
import torch.backends.mps
import numpy as np

from .cnlp_rest import UnannotatedDocument, get_dataset
from ..BaselineModels import CnnSentenceClassifier

app = FastAPI()
model_name = os.getenv('MODEL_PATH')
if model_name is None:
    sys.stderr.write('This REST container requires a MODEL_PATH environment variable\n')
    sys.exit(-1)
device = os.getenv('MODEL_DEVICE', 'auto')
if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

logger = logging.getLogger('CNN_REST_Processor')
logger.setLevel(logging.DEBUG)

@app.on_event("startup")
async def startup_event():
    conf_file = join(model_name, 'config.json')
    with open(conf_file, 'rt') as fp:
        conf_dict = json.load(fp)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels_dict = {task:len(values) for task,values in conf_dict['label_dictionary'].items()}
    model = CnnSentenceClassifier(len(tokenizer), 
                                  task_names = conf_dict['task_names'], 
                                  num_labels_dict=num_labels_dict, 
                                  embed_dims = conf_dict['cnn_embed_dim'], 
                                  num_filters=conf_dict['num_filters'], 
                                  filters=conf_dict['filters'])
    model.load_state_dict(torch.load(join(model_name, 'pytorch_model.bin')), map_location=torch.device(device))
    
    app.state.model = model.to(device)
    app.state.tokenizer = tokenizer
    app.state.conf_dict = conf_dict

@app.post("/cnn/classify")
async def process(doc: UnannotatedDocument):
    results = []
    instances = [doc.doc_text]
    dataset = get_dataset(instances, app.state.tokenizer, max_length=app.state.conf_dict['max_seq_length'])
    _, logits = app.state.model.forward(input_ids=torch.LongTensor(dataset['input_ids']).to(device),
                                        attention_mask = torch.LongTensor(dataset['attention_mask']).to(device),
                                     )
    
    prediction = int(np.argmax(logits[0].cpu().detach().numpy(), axis=1))
    result = app.state.conf_dict['label_dictionary'][app.state.conf_dict['task_names'][0]][prediction]
    return {'result': result}

def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for serving CNN model outputs.')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("cnlpt.api.cnn_rest:app", host='0.0.0.0', port=args.port, reload=False)


if __name__ == '__main__':
    rest()

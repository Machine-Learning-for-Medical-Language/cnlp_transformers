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

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.data.processors.utils import InputFeatures, InputExample
from torch.utils.data.dataset import Dataset
from transformers.data.processors.glue import glue_convert_examples_to_features
import numpy as np

from CnlpRobertaForClassification import CnlpRobertaForClassification

from seqeval.metrics.sequence_labeling import get_entities

import logging
from time import time

app = FastAPI()
model_name = "tmills/clinical_tempeval"
logger = logging.getLogger('Temporal_REST_Processor')
logger.setLevel(logging.INFO)

labels = ["-1", "1"]
timex_label_list = ["B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME",
                    "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME",
                    "O"]
timex_label_dict = { val:ind for ind,val in enumerate(timex_label_list)}
event_label_list = ["B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
    ,"I-BEFORE/OVERLAP","I-OVERLAP","O"]
event_label_dict = { val:ind for ind,val in enumerate(event_label_list)}

max_length = 128

class TokenizedSentenceDocument(BaseModel):
    '''sent_tokens: a list of sentences, where each sentence is a list of tokens'''
    sent_tokens: List[List[str]]

class Timex(BaseModel):
    begin: int
    end: int
    timeClass: str

class Event(BaseModel):
    begin: int
    end: int
    dtr: str

class TemporalResults(BaseModel):
    ''' statuses: dictionary from entity id to classification decision about negation; true -> negated, false -> not negated'''
    timexes: List[List[Timex]]
    events: List[List[Event]]

class TemporalDocumentDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.timex_label_list = ["B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME",
                                 "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME",
                                 "O"]
        self.event_label_list = ["B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
            ,"I-BEFORE/OVERLAP","I-OVERLAP","O"]
    def __len__(self):
        return len(self.features)
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
    def get_labels(self):
        return [self.timex_label_list, self.event_label_list]
    @classmethod
    def from_instance_list(cls, inst_list, tokenizer):
        examples = []
        for (ind,inst) in enumerate(inst_list):
            guid = 'instance-%d' % (ind)
            examples.append(InputExample(guid=guid, text_a=inst, text_b='', label=None))
        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list = labels,
            output_mode='classification'
        )
        return cls(features)

def create_instance_string(tokens: List[str]):
    return ' '.join(tokens)

@app.on_event("startup")
async def startup_event():
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', '128', '--do_predict']
    # training_args = parserTrainingArguments('save_run/')
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses(args=args)

    app.training_args = training_args

    # training_args.per_device_eval_size = 32
    logger.warn("Eval batch size is: " + str(training_args.eval_batch_size))

@app.post("/temporal/initialize")
async def initialize():
    ''' Load the model from disk and move to the device'''
    config = AutoConfig.from_pretrained(model_name)
    app.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  config=config)
    model = CnlpRobertaForClassification.from_pretrained(model_name, config=config, tagger=[True,True], num_labels_list=[13,9], )
    model.to('cuda')

    app.trainer = Trainer(
        model=model,
        args=app.training_args,
        compute_metrics=None,
    )

@app.post("/temporal/process")
async def process(doc: TokenizedSentenceDocument):
    sents = doc.sent_tokens

    logger.warn('Received document with %d sentences' % (len(sents)))
    instances = []
    start_time = time()

    for sent_ind, token_list in enumerate(sents):
        # logger.debug('Entity ind: %d has offsets (%d, %d)' % (ent_ind, offsets[0], offsets[1]))
        inst_str = create_instance_string(token_list)
        logger.debug('Instance string is %s' % (inst_str))
        instances.append(inst_str)

    dataset = TemporalDocumentDataset.from_instance_list(instances, app.tokenizer)
    preproc_end = time()

    output = app.trainer.predict(test_dataset=dataset)

    timex_predictions = np.argmax(output.predictions[0], axis=2)
    event_predictions = np.argmax(output.predictions[1], axis=2)

    pred_end = time()


    timex_results = []
    event_results = []

    for sent_ind in range(len(dataset)):
        tokens = app.tokenizer.convert_ids_to_tokens(dataset.features[sent_ind].input_ids)
        timex_labels = []
        event_labels = []
        for token_ind in range(1,len(tokens)):
            if dataset[sent_ind].input_ids[token_ind] <= 2:
                break
            if tokens[token_ind].startswith('Ġ'):
                timex_labels.append(timex_label_list[timex_predictions[sent_ind][token_ind]])
                event_labels.append(event_label_list[event_predictions[sent_ind][token_ind]])

        timex_entities = get_entities(timex_labels)
        logging.info("Extracted %d timex entities from the sentence" % (len(timex_entities)))
        timex_results.append( [Timex(timeClass=label[0], begin=label[1], end=label[2]) for label in timex_entities] )

        event_entities = get_entities(event_labels)
        logging.info("Extracted %d events from the sentence" % (len(event_entities)))
        event_results.append( [Event(dtr=label[0], begin=label[1], end=label[2]) for label in event_entities] )

    results = TemporalResults(timexes=timex_results, events=event_results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))

    return results


@app.post("/temporal/collection_process_complete")
async def collection_process_complete():
    app.trainer = None

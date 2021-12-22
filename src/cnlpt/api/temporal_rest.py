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
from ..CnlpRobertaForClassification import CnlpRobertaForClassification
from seqeval.metrics.sequence_labeling import get_entities
import logging
from time import time
from nltk.tokenize import wordpunct_tokenize as tokenize

app = FastAPI()
model_name = "tmills/clinical_tempeval"
logger = logging.getLogger('Temporal_REST_Processor')
logger.setLevel(logging.INFO)

labels = ["-1", "1"]
timex_label_list = ["O", "B-DATE","B-DURATION","B-PREPOSTEXP","B-QUANTIFIER","B-SET","B-TIME","B-SECTIONTIME","B-DOCTIME",
                    "I-DATE","I-DURATION","I-PREPOSTEXP","I-QUANTIFIER","I-SET","I-TIME","I-SECTIONTIME","I-DOCTIME"]
timex_label_dict = { val:ind for ind,val in enumerate(timex_label_list)}
event_label_list = ["O", "B-AFTER","B-BEFORE","B-BEFORE/OVERLAP","B-OVERLAP","I-AFTER","I-BEFORE"
    ,"I-BEFORE/OVERLAP","I-OVERLAP"]
event_label_dict = { val:ind for ind,val in enumerate(event_label_list)}

relation_label_list = ['None', 'CONTAINS']
relation_label_dict = { val:ind for ind,val in enumerate(relation_label_list)}

max_length = 128

class SentenceDocument(BaseModel):
    sentence: str

class TokenizedSentenceDocument(BaseModel):
    '''sent_tokens: a list of sentences, where each sentence is a list of tokens'''
    sent_tokens: List[List[str]]
    metadata: str

class Timex(BaseModel):
    begin: int
    end: int
    timeClass: str

class Event(BaseModel):
    begin: int
    end: int
    dtr: str

class Relation(BaseModel):
    arg1: str
    arg2: str
    category: str

class TemporalResults(BaseModel):
    ''' statuses: dictionary from entity id to classification decision about negation; true -> negated, false -> not negated'''
    timexes: List[List[Timex]]
    events: List[List[Event]]
    relations: List[List[Relation]]

class TemporalDocumentDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

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
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', '8', '--do_predict']
    # training_args = parserTrainingArguments('save_run/')
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses(args=args)

    app.training_args = training_args

    # training_args.per_device_eval_size = 32
    logger.warn("Eval batch size is: " + str(training_args.eval_batch_size))

@app.post("/temporal/initialize")
async def initialize():
    ''' Load the model from disk and move to the device'''
    #import pdb; pdb.set_trace()

    config = AutoConfig.from_pretrained(model_name)
    app.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  config=config)
    model = CnlpRobertaForClassification.from_pretrained(model_name, config=config, tagger=[True,True, False], relations=[False,False,True], num_labels_list=[17, 9, 2], num_attention_heads=256)
    model.to('cuda')

    app.trainer = Trainer(
        model=model,
        args=app.training_args,
        compute_metrics=None,
    )

@app.post("/temporal/process")
async def process(doc: TokenizedSentenceDocument):
    return process_tokenized_sentence_document(doc)

@app.post("/temporal/process_sentence")
async def process_sentence(doc: SentenceDocument):
    tokenized_sent = tokenize(doc.sentence)
    doc = TokenizedSentenceDocument(sent_tokens=[tokenized_sent,], metadata='Single sentence')
    return process_tokenized_sentence_document(doc)

def process_tokenized_sentence_document(doc: TokenizedSentenceDocument):
    sents = doc.sent_tokens
    metadata = doc.metadata

    logger.warn('Received document labeled %s with %d sentences' % (metadata, len(sents)))
    instances = []
    start_time = time()

    for sent_ind, token_list in enumerate(sents):
        inst_str = create_instance_string(token_list)
        logger.debug('Instance string is %s' % (inst_str))
        instances.append(inst_str)

    dataset = TemporalDocumentDataset.from_instance_list(instances, app.tokenizer)
    preproc_end = time()

    output = app.trainer.predict(test_dataset=dataset)

    timex_predictions = np.argmax(output.predictions[0], axis=2)
    event_predictions = np.argmax(output.predictions[1], axis=2)
    rel_predictions = np.argmax(output.predictions[2], axis=3)
    rel_inds = np.where(rel_predictions > 0)
    logging.debug('Found relation indices: %s' % (str(rel_inds)))

    rels_by_sent = {}
    for rel_num in range(len(rel_inds[0])):
        sent_ind = rel_inds[0][rel_num]
        if not sent_ind in rels_by_sent:
            rels_by_sent[sent_ind] = []

        arg1_ind = rel_inds[1][rel_num]
        arg2_ind = rel_inds[2][rel_num]
        rel_cat = rel_predictions[sent_ind,arg1_ind,arg2_ind]

        rels_by_sent[sent_ind].append( (arg1_ind, arg2_ind, rel_cat) )

    pred_end = time()

    timex_results = []
    event_results = []
    rel_results = []

    for sent_ind in range(len(dataset)):
        tokens = app.tokenizer.convert_ids_to_tokens(dataset.features[sent_ind].input_ids)
        wpind_to_ind = {}
        timex_labels = []
        event_labels = []
        for token_ind in range(1,len(tokens)):
            if dataset[sent_ind].input_ids[token_ind] <= 2:
                break
            if tokens[token_ind].startswith('Ġ'):
                wpind_to_ind[token_ind] = len(wpind_to_ind)
                timex_labels.append(timex_label_list[timex_predictions[sent_ind][token_ind]])
                event_labels.append(event_label_list[event_predictions[sent_ind][token_ind]])

        timex_entities = get_entities(timex_labels)
        logging.info("Extracted %d timex entities from the sentence" % (len(timex_entities)))
        timex_results.append( [Timex(timeClass=label[0], begin=label[1], end=label[2]) for label in timex_entities] )

        event_entities = get_entities(event_labels)
        logging.info("Extracted %d events from the sentence" % (len(event_entities)))
        event_results.append( [Event(dtr=label[0], begin=label[1], end=label[2]) for label in event_entities] )

        rel_sent_results = []
        for rel in rels_by_sent.get(sent_ind, []):
            arg1 = None
            arg2 = None
            if rel[0] not in wpind_to_ind or rel[1] not in wpind_to_ind:
                logging.warn('Found a relation to a non-leading wordpiece token... ignoring')
                continue

            arg1_ind = wpind_to_ind[rel[0]]
            arg2_ind = wpind_to_ind[rel[1]]

            sent_timexes = timex_results[-1]
            for timex_ind,timex in enumerate(sent_timexes):
                if timex.begin == arg1_ind:
                    arg1 = 'TIMEX-%d' % timex_ind
                if timex.begin == arg2_ind:
                    arg2 = 'TIMEX-%d' % timex_ind

            sent_events = event_results[-1]
            for event_ind,event in enumerate(sent_events):
                if event.begin == arg1_ind:
                    arg1 = 'EVENT-%d' % event_ind
                if event.begin == arg2_ind:
                    arg2 = 'EVENT-%d' % event_ind

            if arg1 is None or arg2 is None:
                logging.warn('Tried to build a relation but couldn\'t align one of the arguments: %s to words %s [null=%s], %s [null=%s]' % (str(rel), sents[sent_ind][arg1_ind], str(arg1 is None), sents[sent_ind][arg2_ind], str(arg2 is None)))
                continue

            rel = Relation(arg1=arg1, arg2=arg2, category=relation_label_list[rel[2]])
            rel_sent_results.append( rel )


        rel_results.append(rel_sent_results)

    results = TemporalResults(timexes=timex_results, events=event_results, relations=rel_results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))

    return results


@app.post("/temporal/collection_process_complete")
async def collection_process_complete():
    app.trainer = None


def rest():
    import argparse

    parser = argparse.ArgumentParser(description='Run the http server for temporal')
    parser.add_argument('-p', '--port', type=int, help='The port number to run the server on', default=8000)

    args = parser.parse_args()

    import uvicorn
    uvicorn.run("cnlpt.api.temporal_rest:app", host='0.0.0.0', port=args.port, reload=True)


if __name__ == '__main__':
    rest()

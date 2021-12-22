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

from .temporal_rest import timex_label_list, timex_label_dict, TokenizedSentenceDocument, SentenceDocument, Timex, TemporalResults, TemporalDocumentDataset, create_instance_string

app = FastAPI()
model_name = "tmills/timex-thyme-colon"
logger = logging.getLogger('Timex_REST_Processor')
logger.setLevel(logging.INFO)

max_length = 128

@app.on_event("startup")
async def startup_event():
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', '8', '--do_predict']
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses(args=args)

    app.training_args = training_args

    logger.warn("Eval batch size is: " + str(training_args.eval_batch_size))

@app.post("/temporal/initialize")
async def initialize():
    ''' Load the model from disk and move to the device'''
    #import pdb; pdb.set_trace()

    config = AutoConfig.from_pretrained(model_name)
    app.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  config=config)
    model = CnlpRobertaForClassification.from_pretrained(model_name, config=config, tagger=[True], relations=[False], num_labels_list=[17])
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
    logger.warn('Dataset is as follows: %s' % (str(dataset.features)))

    preproc_end = time()

    output = app.trainer.predict(test_dataset=dataset)

    timex_predictions = np.argmax(output.predictions[0], axis=2)
    timex_results = []
    event_results = []
    relation_results = []

    pred_end = time()
    
    for sent_ind in range(len(dataset)):
        tokens = app.tokenizer.convert_ids_to_tokens(dataset.features[sent_ind].input_ids)
        wpind_to_ind = {}
        timex_labels = []
        for token_ind in range(1,len(tokens)):
            if dataset[sent_ind].input_ids[token_ind] <= 2:
                break
            if tokens[token_ind].startswith('Ġ'):
                wpind_to_ind[token_ind] = len(wpind_to_ind)
                timex_labels.append(timex_label_list[timex_predictions[sent_ind][token_ind]])

        timex_entities = get_entities(timex_labels)
        logging.info("Extracted %d timex entities from the sentence" % (len(timex_entities)))
        timex_results.append( [Timex(timeClass=label[0], begin=label[1], end=label[2]) for label in timex_entities] )
        event_results.append( [] )
        relation_results.append( [] )


    results = TemporalResults(timexes=timex_results, events=event_results, relations=relation_results)

    postproc_end = time()

    preproc_time = preproc_end - start_time
    pred_time = pred_end - preproc_end
    postproc_time = postproc_end - pred_end

    logging.info("Pre-processing time: %f, processing time: %f, post-processing time %f" % (preproc_time, pred_time, postproc_time))

    return results


@app.post("/temporal/collection_process_complete")
async def collection_process_complete():
    app.trainer = None

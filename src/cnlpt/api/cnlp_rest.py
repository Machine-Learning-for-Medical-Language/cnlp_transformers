
# Core python imports
import os

# FastAPI imports
from pydantic import BaseModel
from typing import List

# Modeling imports
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import logging

# intra-library imports
from ..CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from ..cnlp_data import cnlp_preprocess_data

class EntityDocument(BaseModel):
    ''' doc_text: The raw text of the document
    offset:  A list of entities, where each is a tuple of character offsets into doc_text for that entity'''
    doc_text: str
    entities: List[List[int]]

def get_dataset(inst_list, tokenizer, label_lists, tasks, max_length=128):
    dataset = Dataset.from_dict({'text':inst_list})
    task_dataset = dataset.map(
                    cnlp_preprocess_data,
                    batched=True,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset, organizing labels, creating hierarchical segments if necessary",
                    batch_size=100,
                    fn_kwargs = {
                        'tokenizer':tokenizer,
                        'max_length':max_length,
                        'label_lists':label_lists,
                        'inference': True,
                        # TODO: need to get this from the model if necessary
                        #'hierarchical':self.hierarchical,
                        # 'chunk_len':self.args.chunk_len,
                        # 'num_chunks':self.args.num_chunks,
                        # 'insert_empty_chunk_at_beginning':self.args.insert_empty_chunk_at_beginning,
                        'truncate_examples': True,
                        'tasks': tasks,
                }
    )
    return task_dataset

def create_instance_string(doc_text: str, offsets : List[int]):
    start = max(0, offsets[0]-100)
    end = min(len(doc_text), offsets[1]+100)
    raw_str = doc_text[start:offsets[0]] + ' <e> ' + doc_text[offsets[0]:offsets[1]] + ' </e> ' + doc_text[offsets[1]:end]
    return raw_str.replace('\n', ' ')

def initialize_cnlpt_model(app, model_name, cuda=True, batch_size=8):
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', str(batch_size), '--do_predict', '--report_to', 'none']
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses(args=args)

    app.state.training_args = training_args

    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    config = AutoConfig.from_pretrained(model_name)
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  config=config)
    model = CnlpModelForClassification.from_pretrained(model_name, cache_dir=os.getenv('HF_CACHE'), config=config)
    if cuda and not torch.cuda.is_available():
        logging.warning('CUDA is set to True (probably a default) but was not available; setting to False and proceeding. If you have a GPU you need to debug why pytorch cannot see it.')
        cuda = False
    
    if cuda:
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    app.state.trainer = Trainer(
        model=model,
        args=app.state.training_args,
        compute_metrics=None,
    )

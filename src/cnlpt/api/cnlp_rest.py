
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
from transformers.data.processors.utils import InputFeatures, InputExample
from torch.utils.data.dataset import Dataset
import torch
import logging

# intra-library imports
from ..CnlpModelForClassification import CnlpModelForClassification, CnlpConfig
from ..cnlp_data import cnlp_convert_examples_to_features

class EntityDocument(BaseModel):
    ''' doc_text: The raw text of the document
    offset:  A list of entities, where each is a tuple of character offsets into doc_text for that entity'''
    doc_text: str
    entities: List[List[int]]

class ClassificationDocumentDataset(Dataset):
    def __init__(self, features, label_list):
        self.features = features
        self.label_list = label_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    @classmethod
    def from_instance_list(cls, inst_list, tokenizer, label_list, max_length=128):
        examples = []
        for (ind,inst) in enumerate(inst_list):
            guid = 'instance-%d' % (ind)
            examples.append(InputExample(guid=guid, text_a=inst, text_b='', label=None))

        features = cnlp_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list = label_list,
            output_mode='classification',
            inference=True
        )
        return cls(features, label_list)

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




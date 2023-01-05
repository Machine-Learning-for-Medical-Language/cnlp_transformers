import os
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from cnlpt.CnlpModelForClassification import CnlpModelForClassification, CnlpConfig

def pre_initialize_cnlpt_model(model_name, cuda=True, batch_size=8):
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', str(batch_size), '--do_predict', '--report_to', 'none']
    #parser = HfArgumentParser((TrainingArguments,))
    #training_args, = parser.parse_args_into_dataclasses(args=args)


    AutoConfig.register("cnlpt", CnlpConfig)
    AutoModel.register(CnlpConfig, CnlpModelForClassification)

    print('initializing ' +model_name)
    print('fetching pretrained configs')
    config = AutoConfig.from_pretrained(model_name)
    print('fetching pretrained tokens')
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
    print('fetching pretrained models')
    model = CnlpModelForClassification.from_pretrained(model_name, cache_dir=os.getenv('HF_CACHE'), config=config)

def dtr():
  pre_initialize_cnlpt_model("tmills/tiny-dtr")

def event():
  pre_initialize_cnlpt_model("tmills/event-thyme-colon-pubmedbert")

def negation():
  pre_initialize_cnlpt_model("tmills/cnlpt-negation-roberta-sharpseed")

def temporal():
  pre_initialize_cnlpt_model("tmills/thyme1-e2e")

def timex():
  pre_initialize_cnlpt_model("tmills/timex-thyme-colon-pubmedbert")
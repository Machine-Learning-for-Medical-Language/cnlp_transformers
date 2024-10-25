import os

from transformers import AutoConfig, AutoTokenizer

from cnlpt.CnlpModelForClassification import CnlpModelForClassification
from cnlpt.HierarchicalTransformer import HierarchicalModel
from cnlpt.train_system import is_hub_model


def pre_initialize_cnlpt_model(model_name, cuda=True, batch_size=8):
    print("initializing " + model_name)
    print("fetching pretrained configs")
    config = AutoConfig.from_pretrained(model_name)
    print("fetching pretrained tokens")
    AutoTokenizer.from_pretrained(model_name, config=config)
    print("fetching pretrained models")
    CnlpModelForClassification.from_pretrained(
        model_name, cache_dir=os.getenv("HF_CACHE"), config=config
    )


def pre_initialize_hier_model(model_name, cuda=True, batch_size=8):
    print("initializing " + model_name)
    print("fetching pretrained configs")
    config = AutoConfig.from_pretrained(model_name)
    print("fetching pretrained tokens")
    AutoTokenizer.from_pretrained(model_name, config=config)
    print("fetching pretrained models")
    HierarchicalModel.from_pretrained(
        model_name, cache_dir=os.getenv("HF_CACHE"), config=config
    )


def current():
    pre_initialize_cnlpt_model("mlml-chip/current-thyme")


def dtr():
    pre_initialize_cnlpt_model("tmills/tiny-dtr")


def event():
    pre_initialize_cnlpt_model("tmills/event-thyme-colon")


def negation():
    pre_initialize_cnlpt_model("mlml-chip/negation_pubmedbert_sharpseed")


def termexists():
    pre_initialize_cnlpt_model("mlml-chip/termexists_pubmedbert_ssm")


def temporal():
    pre_initialize_cnlpt_model("mlml-chip/thyme2_colon_e2e")


def timex():
    pre_initialize_cnlpt_model("tmills/timex-thyme-colon")


def hier(model_dir):
    if is_hub_model(model_dir):
        pre_initialize_hier_model(model_dir)

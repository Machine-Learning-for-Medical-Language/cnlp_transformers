from pathlib import Path

import pytest
from transformers.models.auto.tokenization_auto import AutoTokenizer

from cnlpt.args import CnlpDataArguments
from cnlpt.new_data.cnlp_dataset import CnlpDataset


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


@pytest.fixture
def dataset_dir():
    return Path(__file__).parent.parent / "common" / "datasets" / "json"


def test_create_cnlp_dataset(tokenizer, dataset_dir):
    args = CnlpDataArguments([str(dataset_dir.resolve())])
    cnlp_dataset = CnlpDataset(args=args, tokenizer=tokenizer, hierarchical=False)
    assert len(cnlp_dataset.train_data) == len(cnlp_dataset.validation_data) == 4

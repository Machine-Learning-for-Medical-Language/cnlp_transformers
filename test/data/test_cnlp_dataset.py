import numpy as np

from cnlpt.args import CnlpDataArguments
from cnlpt.data import CnlpDataset

from ..common.fixtures import random_cnlp_data_options


@random_cnlp_data_options(
    tasks=[("classification", 3), ("tagging", 3), ("relations", 3)],
    n_train=5,
    n_test=6,
    n_dev=7,
)
def test_create_random_dataset(tokenizer, random_cnlp_data_dir):
    args = CnlpDataArguments([random_cnlp_data_dir])
    cnlp_dataset = CnlpDataset(args=args, tokenizer=tokenizer, hierarchical=False)
    assert len(cnlp_dataset.train_data) == 5
    assert len(cnlp_dataset.test_data) == 6
    assert len(cnlp_dataset.validation_data) == 7


@random_cnlp_data_options(
    tasks=[("classification", 3), ("classification", 3)], n_train=3
)
def test_labels_shape_classification_only(tokenizer, random_cnlp_data_dir):
    """When we don't load tagging or relations tasks, there is no need for the
    sequence length dimension in the labels. This test ensures that the preprocessing
    generates labels with the shape (batch, n_tasks).
    """
    batch_size = 3
    max_seq_len = 128

    args = CnlpDataArguments(
        [random_cnlp_data_dir],
        max_seq_length=max_seq_len,
        overwrite_cache=True,
    )
    cnlp_dataset = CnlpDataset(args=args, tokenizer=tokenizer, hierarchical=False)
    batch = next(cnlp_dataset.train_data.iter(batch_size))
    labels = np.array(batch["label"])

    # a single label for each task
    assert labels.shape == (batch_size, 2)


@random_cnlp_data_options(
    tasks=[("classification", 3), ("tagging", 3), ("relations", 3)], n_train=3
)
def test_labels_shape_mixed_tasks(tokenizer, random_cnlp_data_dir):
    """In contrast with the `test_labels_shape_classification_only` test above,
    when we load tagging and/or relations tasks we need a sequence length dimension.
    This test ensures that the preprocessing generates labels with the shape
    (batch, sequence_len, n_classification_tasks + n_tagging_tasks + sequence_len * n_relations_tasks)
    """
    batch_size = 3
    max_seq_len = 128

    args = CnlpDataArguments(
        [random_cnlp_data_dir],
        max_seq_length=max_seq_len,
        overwrite_cache=True,
    )
    cnlp_dataset = CnlpDataset(args=args, tokenizer=tokenizer, hierarchical=False)
    batch = next(cnlp_dataset.train_data.iter(batch_size))
    labels = np.array(batch["label"])

    assert labels.shape == (
        batch_size,
        max_seq_len,
        # one each for classification and tagging tasks, plus max_seq_len for relations
        1 + 1 + max_seq_len,
    )

from typing import Union

import numpy as np
import polars as pl
import pytest

from cnlpt.data.analysis import _rel_matrix_to_rels, make_preds_df

from ..common.fixtures import random_cnlp_data_options


@random_cnlp_data_options(
    tasks=(("classification", 3), ("tagging", 3), ("relations", 3)),
    n_train=5,
    n_test=5,
    n_dev=5,
)
def test_predict(random_cnlp_train_system):
    predictions = random_cnlp_train_system.predict()
    seq_len = random_cnlp_train_system.dataset.max_seq_length
    df = make_preds_df(predictions)
    rel_labels = predictions.task_predictions["task_2_relations"].task.labels
    assert df.schema == pl.Schema(
        {
            "sample_idx": pl.Int64,
            "text": pl.String,
            "word_ids": pl.Array(pl.Int64, shape=(seq_len,)),
            "sample_id": pl.String,
            "task_0_classification": pl.Struct(
                {
                    "labels": pl.Struct({"ids": pl.Int64, "values": pl.String}),
                    "predictions": pl.Struct({"ids": pl.Int64, "values": pl.String}),
                    "model_output": pl.Struct(
                        {
                            "logits": pl.Array(pl.Float32, shape=(3,)),
                            "probs": pl.Array(pl.Float32, shape=(3,)),
                        }
                    ),
                }
            ),
            "task_1_tagging": pl.Struct(
                {
                    "labels": pl.Struct(
                        {
                            "ids": pl.Array(pl.Int64, shape=(seq_len,)),
                            "values": pl.Array(pl.String, shape=(seq_len,)),
                            "spans": pl.List(
                                pl.Struct(
                                    {
                                        "text": pl.String,
                                        "tag": pl.String,
                                        "start": pl.Int64,
                                        "end": pl.Int64,
                                        "valid": pl.Boolean,
                                    }
                                )
                            ),
                        }
                    ),
                    "predictions": pl.Struct(
                        {
                            "ids": pl.Array(pl.Int64, shape=(seq_len,)),
                            "values": pl.Array(pl.String, shape=(seq_len,)),
                            "spans": pl.List(
                                pl.Struct(
                                    {
                                        "text": pl.String,
                                        "tag": pl.String,
                                        "start": pl.Int64,
                                        "end": pl.Int64,
                                        "valid": pl.Boolean,
                                    }
                                )
                            ),
                        }
                    ),
                    "model_output": pl.Struct(
                        {
                            "logits": pl.Array(pl.Float32, shape=(seq_len, 3)),
                            "probs": pl.Array(pl.Float32, shape=(seq_len, 3)),
                        }
                    ),
                }
            ),
            "task_2_relations": pl.Struct(
                {
                    "labels": pl.Struct(
                        {
                            "ids": pl.Array(pl.Int64, shape=(seq_len, seq_len)),
                            "values": pl.Array(pl.String, shape=(seq_len, seq_len)),
                            "relations": pl.List(
                                pl.Struct(
                                    {
                                        "arg1_wid": pl.Int64,
                                        "arg1_text": pl.String,
                                        "arg2_wid": pl.Int64,
                                        "arg2_text": pl.String,
                                        "label": pl.String,
                                    }
                                )
                            ),
                        }
                    ),
                    "predictions": pl.Struct(
                        {
                            "ids": pl.Array(pl.Int64, shape=(seq_len, seq_len)),
                            "values": pl.Array(pl.String, shape=(seq_len, seq_len)),
                            "relations": pl.List(
                                pl.Struct(
                                    {
                                        "arg1_wid": pl.Int64,
                                        "arg1_text": pl.String,
                                        "arg2_wid": pl.Int64,
                                        "arg2_text": pl.String,
                                        "label": pl.String,
                                        "probs": pl.Struct(
                                            {label: pl.Float32 for label in rel_labels}
                                        ),
                                    }
                                )
                            ),
                        }
                    ),
                    "model_output": pl.Struct(
                        {
                            "logits": pl.Array(
                                pl.Float32,
                                shape=(seq_len, seq_len, 3 + 1),  # 3 labels plus "None"
                            ),
                            "probs": pl.Array(
                                pl.Float32,
                                shape=(seq_len, seq_len, 3 + 1),  # 3 labels plus "None"
                            ),
                        }
                    ),
                }
            ),
        }
    )


def _rel_df(text: str, word_ids: list[Union[int, None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"sample_idx": [0], "text": [text], "word_ids": [word_ids]},
        schema_overrides={"word_ids": pl.Array(pl.Int64, shape=len(word_ids))},
    )


def test_rel_matrix_to_rels_probs():
    # [CLS] aa bb cc [SEP]
    df = _rel_df("aa bb cc", [None, 0, 1, 2, None])
    labels = ["None", "R1", "R2"]
    matrix = np.full((1, 5, 5), "None", dtype=object)
    matrix[0, 1, 3] = "R2"
    probs = np.zeros((1, 5, 5, 3), dtype=np.float32)
    probs[0, 1, 3] = [0.25, 0.25, 0.5]

    relations = _rel_matrix_to_rels(df, matrix, probs=probs, labels=labels)[
        "relations"
    ].to_list()

    assert relations == [
        [
            {
                "arg1_wid": 0,
                "arg1_text": "aa",
                "arg2_wid": 2,
                "arg2_text": "cc",
                "label": "R2",
                "probs": {"None": 0.25, "R1": 0.25, "R2": 0.5},
            }
        ]
    ]


def test_rel_matrix_to_rels_ignores_continuation_subwords():
    # [CLS] aa ##a bb [SEP] -- the model is only trained on first-subword positions
    df = _rel_df("aa bb", [None, 0, 0, 1, None])
    matrix = np.full((1, 5, 5), "None", dtype=object)
    matrix[0, 1, 3] = "R1"
    matrix[0, 2, 3] = "R2"

    relations = _rel_matrix_to_rels(df, matrix)["relations"].to_list()

    assert relations == [
        [
            {
                "arg1_wid": 0,
                "arg1_text": "aa",
                "arg2_wid": 1,
                "arg2_text": "bb",
                "label": "R1",
            }
        ]
    ]


def test_rel_matrix_to_rels_requires_labels_with_probs():
    df = _rel_df("aa", [None, 0, None])
    matrix = np.full((1, 3, 3), "None", dtype=object)
    with pytest.raises(ValueError):
        _rel_matrix_to_rels(df, matrix, probs=np.zeros((1, 3, 3, 1)))

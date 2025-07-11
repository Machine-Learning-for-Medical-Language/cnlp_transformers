import polars as pl

from cnlpt.data.analysis import make_preds_df

from ..common.fixtures import random_cnlp_data_options


@random_cnlp_data_options(
    tasks=(("classification", 3), ("tagging", 3), ("relations", 3)),
    n_train=5,
    n_test=5,
    n_dev=5,
)
def test_predict(random_cnlp_train_system):
    predictions = random_cnlp_train_system.predict()
    seq_len = random_cnlp_train_system.data_args.max_seq_length
    df = make_preds_df(predictions)
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
                        }
                    ),
                    "predictions": pl.Struct(
                        {
                            "ids": pl.Array(pl.Int64, shape=(seq_len, seq_len)),
                            "values": pl.Array(pl.String, shape=(seq_len, seq_len)),
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

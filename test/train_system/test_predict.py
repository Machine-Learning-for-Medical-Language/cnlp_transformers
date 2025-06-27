import polars as pl

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

    df = predictions.to_data_frame(include_logits=True, include_probs=True)
    assert df.schema == pl.Schema(
        {
            "sample_idx": pl.Int64,
            "sample_id": pl.String,
            "text": pl.String,
            "tokens": pl.Array(pl.String, shape=(seq_len,)),
            "task_0_classification": pl.Struct(
                {
                    "raw_label": pl.String,
                    "label": pl.Float32,
                    "predicted_label": pl.Int64,
                    "logits": pl.Array(pl.Float32, shape=(3,)),
                    "probs": pl.Array(pl.Float32, shape=(3,)),
                }
            ),
            "task_1_tagging": pl.Struct(
                {
                    "raw_label": pl.String,
                    "label": pl.Array(pl.Float32, shape=(seq_len,)),
                    "predicted_label": pl.Array(pl.Int64, shape=(seq_len,)),
                    "logits": pl.Array(pl.Float32, shape=(seq_len, 3)),
                    "probs": pl.Array(pl.Float32, shape=(seq_len, 3)),
                }
            ),
            "task_2_relations": pl.Struct(
                {
                    "raw_label": pl.String,
                    "label": pl.Array(pl.Float32, shape=(seq_len, seq_len)),
                    "predicted_label": pl.Array(pl.Int64, shape=(seq_len, seq_len)),
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
    )

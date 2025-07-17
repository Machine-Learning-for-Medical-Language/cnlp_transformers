from typing import Any

import numpy as np
import polars as pl

from .predictions import CnlpPredictions
from .task_info import CLASSIFICATION, RELATIONS, TAGGING


def _bio_tags_to_spans(df: pl.DataFrame, tags_col: pl.Expr):
    """
    Convert BIO-tagged data to labeled spans.
    """
    return (
        df.lazy()
        .select(
            "sample_idx",
            "text",
            "word_ids",
            tags=tags_col,
            # keep track of each token's inner index within the sample
            inner_index=pl.int_ranges(pl.col("word_ids").arr.len()),
        )
        # one row per token+tag
        .explode("inner_index", "word_ids", "tags")
        # remove special and masked tokens
        .filter(pl.col("word_ids").is_not_null())
        .unique(["sample_idx", "word_ids"], keep="first")
        # group by sample
        .group_by("sample_idx", "text")
        .agg(
            pl.col("inner_index", "word_ids", "tags").sort_by("inner_index"),
            # span ID changes whenever we see a "B" tag, an "O" tag, or a tag with a new label
            span_id=pl.struct(
                # b_id increments every time we see a "B" tag
                b_id=pl.col("tags")
                .sort_by("inner_index")
                .str.starts_with("B")
                .cum_sum(),
                # o_id increments every time we see an "O" tag
                o_id=pl.col("tags").sort_by("inner_index").eq(pl.lit("O")).cum_sum(),
                # label_id is unique for each label category
                label_id=pl.col("tags").sort_by("inner_index").str.slice(1),
            ),
        )
        # one row per token again, now with span IDs
        .explode("inner_index", "word_ids", "tags", "span_id")
        # remove tokens with "O" tags
        .filter(pl.col("tags").ne(pl.lit("O")))
        # group by span
        .group_by("sample_idx", "text", "span_id")
        .agg(
            first_tag=pl.col("tags").sort_by("inner_index").first(),
            words=pl.col("text")
            .str.split(pl.lit(" "), inclusive=True)
            .list.get(pl.col("word_ids").sort()),
            start=pl.col("word_ids").min(),
            end=pl.col("word_ids").max(),
        )
        .drop("span_id")
        .with_columns(
            # get label from tag
            tag=pl.col("first_tag").str.split(pl.lit("-")).list.last(),
            # join tokens to get span text
            text=pl.col("words").list.join(pl.lit("")),
            # the span is invalid if the first tag is not a "B" tag
            valid=pl.col("first_tag").str.starts_with(pl.lit("B")),
        )
        # group by sample
        .group_by("sample_idx")
        # collect span data into structs, sorted by their order in the sample
        .agg(spans=pl.struct("text", "tag", "start", "end", "valid").sort_by("start"))
        .collect()
    )


# TODO(ian) This is not very fast! I experimented with pure polars and pure numpy approaches,
# and this hybrid approach was faster than either. If we end up needing to optimize this,
# maybe multiprocessing can help? (i.e., do multiple rows simultaneously)
# There might also be a polars-only approach where we break the dataframe into small chunks first,
# avoiding the memory bottleneck that arises when exploding the matrix row.
def _rel_matrix_to_rels(df: pl.DataFrame, matrix_col: pl.Expr):
    def extract_relations(input) -> list[dict[str, Any]]:
        matrix, word_ids, text = input["matrix"], input["word_ids"], input["text"]

        if matrix is None or word_ids is None:
            return []

        arr = np.array(matrix)
        word_ids_arr = np.array(word_ids)

        words = np.array(text)

        rel_idxs = (
            (arr != "None") & (arr != "[MASK]") & ~np.eye(arr.shape[0]).astype(bool)
        )
        rel_positions = np.argwhere(rel_idxs)

        if len(rel_positions) == 0:
            return []

        def remove_nulls(arr, fill):
            return np.where(arr == None, fill, arr)  # noqa: E711

        arg1_wids = word_ids_arr[rel_positions[:, 0]]
        arg1_words = np.where(
            arg1_wids == None,  # noqa: E711
            None,
            words[remove_nulls(arg1_wids, 0).astype(int)],
        )

        arg2_wids = word_ids_arr[rel_positions[:, 1]]
        arg2_words = np.where(
            arg2_wids == None,  # noqa: E711
            None,
            words[remove_nulls(arg2_wids, 0).astype(int)],
        )

        labels = arr[rel_positions[:, 0], rel_positions[:, 1]]

        valid_word_mask = (arg1_wids != None) & (arg2_wids != None)  # noqa: E711

        return [
            {
                "arg1_wid": int(wida),
                "arg1_text": wa,
                "arg2_wid": int(widb),
                "arg2_text": wb,
                "label": label,
            }
            for wida, wa, widb, wb, label in zip(
                arg1_wids[valid_word_mask],
                arg1_words[valid_word_mask],
                arg2_wids[valid_word_mask],
                arg2_words[valid_word_mask],
                labels[valid_word_mask],
            )
        ]

    return df.select(
        "sample_idx",
        relations=pl.struct(
            "word_ids", matrix=matrix_col, text=pl.col("text").str.split(" ")
        ).map_elements(
            extract_relations,
            strategy="threading",
            return_dtype=pl.List(
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
        ),
    )


def make_preds_df(
    predictions: CnlpPredictions,
    *task_names: str,
):
    """Create a polars DataFrame for analysis from a CnlpPredictions instance.

    For relations tasks, this will likely be slow! Converting the model's output matrix
    into relations is expensive.

    Args:
        predictions: The CnlpPredictions instance.
        task_names: One or more tasks to include in the analysis. If omitted, all tasks will be included.

    Returns:
        The DataFrame for analysis.
    """
    seq_len = len(predictions.input_data["input_ids"][0])

    df_data = {
        "sample_idx": list(range(len(predictions.input_data))),
        "text": predictions.input_data["text"],
        "word_ids": predictions.input_data["word_ids"],
    }

    if "id" in predictions.input_data.column_names:
        df_data |= {"sample_id": predictions.input_data["id"]}

    df = pl.DataFrame(
        df_data, schema_overrides={"word_ids": pl.Array(pl.Int64, shape=seq_len)}
    )

    if len(task_names) > 0:
        tasks = [predictions.task_predictions[tn].task for tn in task_names]
    else:
        tasks = predictions.tasks

    for task in tasks:
        task_pred = predictions.task_predictions[task.name]
        df = df.with_columns(
            pl.struct(
                labels=pl.struct(
                    ids=task_pred.labels,
                    values=task_pred.target_str_labels,
                ),
                predictions=pl.struct(
                    ids=task_pred.predicted_int_labels,
                    values=task_pred.predicted_str_labels,
                ),
                model_output=pl.struct(
                    logits=task_pred.logits,
                    probs=task_pred.probs,
                ),
            ).alias(task.name)
        )

        if task.type == CLASSIFICATION:
            # classification output is already pretty human-interpretable
            pass
        elif task.type == TAGGING:
            # for tagging, we'll convert BIO tags to labeled spans
            df = df.join(
                _bio_tags_to_spans(
                    df, pl.col(task.name).struct.field("labels").struct.field("values")
                ),
                on="sample_idx",
                how="left",
            ).rename({"spans": "target_spans"})

            df = df.join(
                _bio_tags_to_spans(
                    df,
                    pl.col(task.name)
                    .struct.field("predictions")
                    .struct.field("values"),
                ),
                on="sample_idx",
                how="left",
            ).rename({"spans": "predicted_spans"})

            df = df.with_columns(
                pl.col(task.name).struct.with_fields(
                    pl.field("labels").struct.with_fields(spans="target_spans"),
                    pl.field("predictions").struct.with_fields(spans="predicted_spans"),
                )
            ).drop("target_spans", "predicted_spans")
        elif task.type == RELATIONS:
            df = df.join(
                _rel_matrix_to_rels(
                    df, pl.col(task.name).struct.field("labels").struct.field("values")
                ),
                on="sample_idx",
                how="left",
            ).rename({"relations": "target_relations"})

            df = df.join(
                _rel_matrix_to_rels(
                    df,
                    pl.col(task.name)
                    .struct.field("predictions")
                    .struct.field("values"),
                ),
                on="sample_idx",
                how="left",
            ).rename({"relations": "predicted_relations"})

            df = df.with_columns(
                pl.col(task.name).struct.with_fields(
                    pl.field("labels").struct.with_fields(relations="target_relations"),
                    pl.field("predictions").struct.with_fields(
                        relations="predicted_relations"
                    ),
                )
            ).drop("target_relations", "predicted_relations")
        else:
            raise ValueError(f"unknown task type {task.type}")

    return df

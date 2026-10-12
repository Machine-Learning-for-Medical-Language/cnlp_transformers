from typing import Any, Union

import numpy as np
import numpy.typing as npt
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


def _first_subword_mask(word_ids: npt.NDArray) -> npt.NDArray:
    """Positions holding the first subword of a word (mirrors relation label encoding in preprocess)."""
    is_word = word_ids != None  # noqa: E711
    prev = np.concatenate([[None], word_ids[:-1]])
    return is_word & (word_ids != prev)


def _relations_dtype(labels: Union[list[str], None]) -> pl.List:
    fields: dict[str, pl.DataType] = {
        "arg1_wid": pl.Int64(),
        "arg1_text": pl.String(),
        "arg2_wid": pl.Int64(),
        "arg2_text": pl.String(),
        "label": pl.String(),
    }
    if labels is not None:
        fields["probs"] = pl.Struct({label: pl.Float32() for label in labels})
    return pl.List(pl.Struct(fields))


def _rel_matrix_to_rels(
    df: pl.DataFrame,
    label_matrices: npt.NDArray,
    probs: Union[npt.NDArray, None] = None,
    labels: Union[list[str], None] = None,
) -> pl.DataFrame:
    """Convert per-token-pair relation label matrices into lists of word-level relations.

    Only first-subword positions are considered, since those are the only positions
    the model is trained on; predictions at continuation subwords would otherwise
    produce duplicate (and possibly conflicting) relations for the same word pair.

    Args:
        df: DataFrame with "sample_idx", "text", and "word_ids" columns.
        label_matrices: String labels of shape (n_samples, seq, seq).
        probs: Optional probabilities of shape (n_samples, seq, seq, n_labels). If given,
            each relation gets a "probs" field mapping every label to its probability.
        labels: Label names in the order of the last axis of `probs`. Required with `probs`.

    Returns:
        DataFrame with "sample_idx" and "relations" columns.
    """
    if (probs is None) != (labels is None):
        raise ValueError("probs and labels must be provided together")

    all_relations: list[list[dict[str, Any]]] = []
    for i, (text, word_ids) in enumerate(df.select("text", "word_ids").iter_rows()):
        wids = np.array(word_ids, dtype=object)
        words = text.split(" ")
        matrix = label_matrices[i]

        valid = _first_subword_mask(wids)
        rel_mask = (
            valid[:, None]
            & valid[None, :]
            & (matrix != "None")
            & (matrix != "[MASK]")
            & ~np.eye(len(wids), dtype=bool)
        )

        sample_relations: list[dict[str, Any]] = []
        for a, b in np.argwhere(rel_mask):
            relation: dict[str, Any] = {
                "arg1_wid": int(wids[a]),
                "arg1_text": words[wids[a]],
                "arg2_wid": int(wids[b]),
                "arg2_text": words[wids[b]],
                "label": str(matrix[a, b]),
            }
            if probs is not None and labels is not None:
                relation["probs"] = dict(zip(labels, probs[i, a, b].tolist()))
            sample_relations.append(relation)
        all_relations.append(sample_relations)

    return pl.DataFrame(
        {"sample_idx": df["sample_idx"], "relations": all_relations},
        schema={"sample_idx": pl.Int64, "relations": _relations_dtype(labels)},
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

    seq_len = len(predictions.input_data["word_ids"][0])

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

    unlabeled = predictions.raw.label_ids is None

    for task in tasks:
        task_pred = predictions.task_predictions[task.name]

        fields = []
        if not unlabeled:
            fields.append(
                pl.struct(
                    ids=task_pred.labels,
                    values=task_pred.target_str_labels,
                ).alias("labels")
            )

        fields.extend(
            [
                pl.struct(
                    ids=task_pred.predicted_int_labels,
                    values=task_pred.predicted_str_labels,
                ).alias("predictions"),
                pl.struct(
                    logits=task_pred.logits,
                    probs=task_pred.probs,
                ).alias("model_output"),
            ]
        )

        df = df.with_columns(pl.struct(fields).alias(task.name))

        if task.type == CLASSIFICATION:
            # classification output is already pretty human-interpretable
            pass
        elif task.type == TAGGING:
            # for tagging, we'll convert BIO tags to labeled spans
            tagging_fields = []
            if not unlabeled:
                df = df.join(
                    _bio_tags_to_spans(
                        df,
                        pl.col(task.name).struct.field("labels").struct.field("values"),
                    ),
                    on="sample_idx",
                    how="left",
                ).rename({"spans": "target_spans"})
                tagging_fields.append(
                    pl.field("labels").struct.with_fields(spans="target_spans")
                )

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

            tagging_fields.append(
                pl.field("predictions").struct.with_fields(spans="predicted_spans")
            )

            df = df.with_columns(
                pl.col(task.name).struct.with_fields(tagging_fields)
            ).drop("target_spans", "predicted_spans", strict=False)
        elif task.type == RELATIONS:
            relations_fields = []
            if not unlabeled:
                df = df.join(
                    _rel_matrix_to_rels(df, task_pred.target_str_labels),
                    on="sample_idx",
                    how="left",
                ).rename({"relations": "target_relations"})
                relations_fields.append(
                    pl.field("labels").struct.with_fields(relations="target_relations")
                )

            df = df.join(
                _rel_matrix_to_rels(
                    df,
                    task_pred.predicted_str_labels,
                    probs=task_pred.probs,
                    labels=task.labels,
                ),
                on="sample_idx",
                how="left",
            ).rename({"relations": "predicted_relations"})
            relations_fields.append(
                pl.field("predictions").struct.with_fields(
                    relations="predicted_relations"
                )
            )

            df = df.with_columns(
                pl.col(task.name).struct.with_fields(relations_fields)
            ).drop("target_relations", "predicted_relations", strict=False)
        else:
            raise ValueError(f"unknown task type {task.type}")

    return df

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


def make_preds_df(
    predictions: CnlpPredictions,
    *task_names: str,
):
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
            # TODO(ian) convert raw relation output to human-readable format
            pass
        else:
            raise ValueError(f"unknown task type {task.type}")

    return df

#!/usr/bin/env python3
"""
Data Download Source:
https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com

Data Source:

Felix Gräßer
Institut für Biomedizinische Technik
Technische Universität Dresden
Dresden, Germany
felix.graesser '@' tu-dresden.de

Surya Kallumadi
Kansas State University
Manhattan, Kansas, USA
surya '@' ksu.edu

Important Notes:
When using this dataset, you agree that you
1) only use the data for research purposes
2) don't use the data for any commerical purposes
3) don't distribute the data to anyone else
4) cite UCI data lab and the source
"""

import csv
import sys
from pathlib import Path

import pandas as pd

TRAIN_FILE = "drugLibTrain_raw.tsv"
TEST_FILE = "drugLibTest_raw.tsv"


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 4:
        return "Low"
    elif rating > 4 and rating < 8:
        return "Medium"
    else:
        return "High"


def remove_newline(review):
    review = review.replace("&#039;", "'")
    review = review.replace("\n", " <cr> ")
    review = review.replace("\r", " <cr> ")
    review = review.replace("\t", " ")
    return review


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[-1])

    # read-in files
    df = pd.read_csv(
        input_path / TRAIN_FILE,
        sep="\t",
        usecols=["rating", "benefitsReview", "sideEffectsReview", "commentsReview"],
        dtype={"benefitsReview": str, "sideEffectsReview": str, "commentsReview": str},
        keep_default_na=False,
    )
    test = pd.read_csv(
        input_path / TEST_FILE,
        sep="\t",
        usecols=["rating", "benefitsReview", "sideEffectsReview", "commentsReview"],
        dtype={"benefitsReview": str, "sideEffectsReview": str, "commentsReview": str},
        keep_default_na=False,
    )

    # split into sentiments categories
    test["sentiment"] = test.rating.apply(to_sentiment)
    df["sentiment"] = df.rating.apply(to_sentiment)

    # remove newlines:
    test["benefits"] = test.benefitsReview.apply(remove_newline)
    df["benefits"] = df.benefitsReview.apply(remove_newline)

    test["sideEffects"] = test.sideEffectsReview.apply(remove_newline)
    df["sideEffects"] = df.sideEffectsReview.apply(remove_newline)

    test["comments"] = test.commentsReview.apply(remove_newline)
    df["comments"] = df.commentsReview.apply(remove_newline)

    # remove quotes
    df["text"] = (
        "Benefits: <cr> "
        + df["benefits"].str.replace('"', "")
        + " <cr> Side effects: <cr> "
        + df["sideEffects"].str.replace('"', "")
        + " <cr> Overall comments: <cr "
        + df["comments"].str.replace('"', "")
    )
    test["text"] = (
        "Benefits: <cr> "
        + test["benefits"].str.replace('"', "")
        + " <cr> Side effects: <cr> "
        + test["sideEffects"].str.replace('"', "")
        + " <cr> Overall comments: <cr "
        + test["comments"].str.replace('"', "")
    )

    # split train and dev into 9:1 ratio
    train = df.sample(frac=0.9, random_state=200)
    dev = df.drop(train.index)

    # select column as desired
    test = test[["sentiment", "text"]]
    train = train[["sentiment", "text"]]
    dev = dev[["sentiment", "text"]]

    # output CSVs
    output_path.mkdir(parents=True, exist_ok=True)
    test.to_csv(
        output_path / "test.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
    )
    train.to_csv(
        output_path / "train.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
    )
    dev.to_csv(
        output_path / "dev.tsv",
        sep="\t",
        encoding="utf-8",
        index=False,
        header=True,
        quoting=csv.QUOTE_NONE,
        escapechar=None,
    )


if __name__ == "__main__":
    main()

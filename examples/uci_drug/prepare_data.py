import io
import zipfile
from pathlib import Path

import polars as pl
import requests

DATASET_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/461/drug+review+dataset+druglib+com.zip"
)
DATA_DIR = Path(__file__).parent / "dataset"


def preprocess_raw_data(unprocessed_path: str):
    return pl.read_csv(unprocessed_path, separator="\t").select(
        id="",
        sentiment=pl.col("rating").map_elements(
            lambda rating: "Negative"
            if rating < 5
            else "Neutral"
            if rating < 8
            else "Positive",
            return_dtype=pl.String,
        ),
        text=(
            pl.concat_str(
                "benefitsReview",
                "sideEffectsReview",
                "commentsReview",
                separator=" ",
            )
            .str.replace_all("\n", " <cr> ")
            .str.replace_all("\r", " <cr> ")
            .str.replace_all("\t", " ")
        ),
    )


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    # Download dataset
    response = requests.get(DATASET_ZIP_URL)
    response.raise_for_status()
    zip = zipfile.ZipFile(io.BytesIO(response.content))
    zip.extractall(DATA_DIR)

    raw_train_file = DATA_DIR / "drugLibTrain_raw.tsv"
    raw_test_file = DATA_DIR / "drugLibTest_raw.tsv"

    # Preprocess raw data
    preprocessed_train_data = preprocess_raw_data(raw_train_file)
    preprocessed_test_data = preprocess_raw_data(raw_test_file)

    # 90/10 split for train and dev
    preprocessed_train_data, preprocessed_dev_data = (
        preprocessed_train_data.iter_slices(int(preprocessed_train_data.shape[0] * 0.9))
    )

    # Write to tsv files
    preprocessed_train_data.write_csv(DATA_DIR / "train.tsv", separator="\t")
    preprocessed_dev_data.write_csv(DATA_DIR / "dev.tsv", separator="\t")
    preprocessed_test_data.write_csv(DATA_DIR / "test.tsv", separator="\t")

    # Delete raw data files
    raw_train_file.unlink()
    raw_test_file.unlink()

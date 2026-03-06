# Drug Review Sentiment Classification

## Jupyter notebook example

See the [example notebook](./uci_drug.ipynb) for a step-by-step walkthrough of
how to use CNLPT to train a model for sentiment classification of drug reviews.

## CLI example

If you prefer, you can instead use the CLI to train the model:

### Download and preprocess the data

Use the [`prepare_data.py`](./prepare_data.py) script to download the data and convert it to CNLPT's data format:

```bash
uv run prepare_data.py
```

> [!TIP] About the dataset:
> This script downloads the
> [*Drug Reviews (Druglib.com)* dataset](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com).
> Please be aware of the terms of use:
>
> > Important Notes:
> >
> > When using this dataset, you agree that you
> >
> > 1) only use the data for research purposes
> > 2) don't use the data for any commerical purposes
> > 3) don't distribute the data to anyone else
> > 4) cite UCI data lab and the source
>
> Here is the dataset's BibTeX citation:
>
> ```bibtex
> @misc{drug_reviews_(druglib.com)_461,
>   author       = {Kallumadi, Surya and Grer, Felix},
>   title        = {{Drug Reviews (Druglib.com)}},
>   year         = {2018},
>   howpublished = {UCI Machine Learning Repository},
>   note         = {{DOI}: https://doi.org/10.24432/C55G6J}
> }
> ```

### Train a model

The following example fine-tunes
[the RoBERTa base model](https://huggingface.co/FacebookAI/roberta-base)
with an added projection layer for classification:

```bash
uv run cnlpt train \
 --model_type proj \
 --encoder roberta-base \
 --data_dir ./dataset \
 --task sentiment \
 --output_dir ./train_output \
 --overwrite_output_dir \
 --do_train --do_eval --do_predict \
 --evals_per_epoch 2 \
 --learning_rate 1e-5 \
 --metric_for_best_model 'sentiment.macro_f1' \
 --load_best_model_at_end \
 --save_strategy best
```

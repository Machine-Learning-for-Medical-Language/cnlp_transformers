# Domain-adaptive pretraining

[Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740) (Gururangan et al., ACL 2020)

## Dataset format

`DaptDataset` expects largely the same dataset format to that used by
`ClinicalNlpDataset`. The main restriction is that there
should be a `text` column; datasets with `text_a` and `text_b` columns
will not be accepted.


## Usage

Use `cnlpt.dapt` for domain-adaptive pretraining on an existing encoder. 

```
$ python -m cnlpt.dapt --help
usage: dapt.py [-h] [--encoder_name ENCODER_NAME]
               [--config_name CONFIG_NAME]
               [--tokenizer_name TOKENIZER_NAME]
               [--output_dir OUTPUT_DIR]
               [--overwrite_output_dir [OVERWRITE_OUTPUT_DIR]]
               [--data_dir DATA_DIR] [--cache_dir CACHE_DIR]
               [--chunk_size CHUNK_SIZE]
               [--mlm_probability MLM_PROBABILITY]
               [--test_size TEST_SIZE] [--seed SEED]
               [--no_eval [NO_EVAL]]

optional arguments:
  -h, --help            show this help message and exit
  --encoder_name ENCODER_NAME
                        Path to pretrained model or model
                        identifier from huggingface.co/models
                        (default: roberta-base)
  --config_name CONFIG_NAME
                        Pretrained config name or path if not the
                        same as model_name (default: None)
  --tokenizer_name TOKENIZER_NAME
                        Pretrained tokenizer name or path if not
                        the same as model_name (default: None)
  --output_dir OUTPUT_DIR
                        Directory path to write trained model to.
                        (default: None)
  --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
                        Overwrite the content of the output
                        directory. Use this to continue training if
                        output_dir points to a checkpoint
                        directory. (default: False)
  --data_dir DATA_DIR   The data dir for domain-adaptive
                        pretraining. (default: None)
  --cache_dir CACHE_DIR
                        Where do you want to store the pretrained
                        models downloaded from s3 (default: None)
  --chunk_size CHUNK_SIZE
                        The chunk size for domain-adaptive
                        pretraining. (default: 128)
  --mlm_probability MLM_PROBABILITY
                        The token masking probability for domain-
                        adaptive pretraining. (default: 0.15)
  --test_size TEST_SIZE
                        The test split proportion for domain-
                        adaptive pretraining. (default: 0.2)
  --seed SEED           The random seed to use for a train/test
                        split for domain-adaptive pretraining
                        (requires --dapt-encoder). (default: 42)
  --no_eval [NO_EVAL]   Don't split into train and test; just
                        pretrain. (default: False)

```

This will save the adapted encoder to the disk at `--output_dir`, where
it can then be passed into `train_system` as `--encoder_name`.

The common idiom will be to use `cnlpt.dapt` on a portion of your
unlabeled data (the task dataset), then run `train_system` using a 
labeled dataset of in-domain data. To evaluate the effectiveness of this 
idiom, you can use an artificially-unlabeled dataset and then evaluate 
the fine-tuned classifier out of `train_system` on the labeled portion
of your task dataset.

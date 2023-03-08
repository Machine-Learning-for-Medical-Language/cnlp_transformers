### Fine-tuning for classification: End-to-end example
1. Download data from [Drug Review Dataset (Drugs.com) Data Set](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) to `data` folder and extract. Pay attention to their terms:
   1. only use the data for research purposes
   2. don't use the data for any commerical purposes
   3. don't distribute the data to anyone else
   4. cite us

2. Run ```python -m cnlpt.data.transform_uci_drug <raw dir> <processed dir>``` to preprocess the data from the extract directory into a new directory. This will create {train,dev,test}.tsv in the processed directory specified, where the sentiment ratings have been collapsed into 3 categories.

3. Fine-tune with something like: 

```python -m cnlpt.train_system --task_name sentiment --data_dir <processed dir> --encoder_name roberta-base --do_train --cache cache/ --output_dir temp/ --overwrite_output_dir --evals_per_epoch 5 --do_eval --num_train_epochs 1 --learning_rate 1e-5 --report_to none```

On our hardware, that command results in eval performance like the following:

```'eval_sentiment': {'acc': 0.8115933044017359, 'f1': [0.8981458951773809, 0.8000984130889407, 0.34115019542155217], 'acc_and_f1': [0.8548695997895583, 0.8058458587453383, 0.5763717499116441], 'recall': [0.9443307408923455, 0.8237082066869301, 0.25352697095435683], 'precision': [0.8562679781015125, 0.7778043530255919, 0.5213310580204779]}```

For a demo of how to run the system in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IVT53DBwFxLKftpIn5iKtF0g4xb9yuxm?usp=sharing)
### Fine-tuning for classification: End-to-end example

1. Download data from [Drug Reviews (Druglib.com) Data Set](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com) to `data` folder and extract. Pay attention to their terms:
   1. only use the data for research purposes
   2. don't use the data for any commerical purposes
   3. don't distribute the data to anyone else
   4. cite us

2. Run ```python examples/uci_drug/transform_uci_drug.py <raw dir> <processed dir>``` to preprocess the data from the extract directory into a new directory. This will create {train,dev,test}.tsv in the processed directory specified, where the sentiment ratings have been collapsed into 3 categories.

3. Fine-tune with something like:

```bash
cnlpt train \
 --data_dir <processed dir> \
 --task_name sentiment \
 --encoder_name roberta-base \
 --do_train \
 --do_eval \
 --cache_dir cache/ \
 --output_dir temp/ \
 --overwrite_output_dir \
 --evals_per_epoch 5 \
 --num_train_epochs 1 \
 --learning_rate 1e-5 \
 --report_to none \
 --metric_for_best_model eval_sentiment.avg_micro_f1 \
 --load_best_model_at_end \
 --save_strategy best
```

On our hardware, that command results in eval performance like the following:
```sentiment = {'acc': 0.7041800643086816, 'f1': [0.7916666666666666, 0.7228915662650603, 0.19444444444444442], 'acc_and_f1': [0.7479233654876741, 0.7135358152868709, 0.449312254376563], 'recall': [0.8216216216216217, 0.8695652173913043, 0.12280701754385964], 'precision': [0.7638190954773869, 0.6185567010309279, 0.4666666666666667]}```

#### Error Analysis for Classification

If you run the above command with the `--error_analysis` flag, you can obtain the `dev` instances for which the model made an erroneous
prediction, organized by their original index in `dev` split, in the `eval_predictions...tsv` file in the `--output_dir` argument.  
For us the first line of this file (after the header) is:

```
        text    sentiment
2       Benefits: <cr> helped aleviate whip lash symptoms <cr> Side effects: <cr> none that i noticed <cr> Overall comments: <cr> i took the medications for the prescribed time and symptoms improved, however, I still have some symptoms which are being treated through physical therapy since the accident was only in December     Ground: Medium Predicted: High

```

The number at the beginning of the line, 2, is the index of the instance in the `dev` split.  The `text` column contains the text of the erroneous instances and the following columns are the tasks provided to the model, in this case, just `sentiment`.  `Ground: Medium Predicted: High` indicates that the provided ground truth label for the instance sentiment is `Medium` but the model predicted `High`.  

#### Human Readable Predictions for Classification

Similarly if you run the above command with `--do_predict` you can obtain human readable predictions for the `test` split, in the `test_predictions...tsv` file.  For us the first line of this file (after the header) is:

```
0       Benefits: <cr> The antibiotic may have destroyed bacteria causing my sinus infection.  But it may also have been caused by a virus, so its hard to say. <cr> Side effects: <cr> Some back pain, some nauseau. <cr> Overall comments: <cr> Took the antibiotics for 14 days. Sinus infection was gone after the 6th day.  Low

```

##### Prediction Probability Outputs for Classification

(Currently only supported for classification tasks), if you run the above command with the `--output_prob` flag, you can see the model's softmax-obtained probability for the predicted classification label.  The first error analysis sample from `dev` would now looks like:

```
 text sentiment
2       Benefits: <cr> helped aleviate whip lash symptoms <cr> Side effects: <cr> none that i noticed <cr> Overall comments: <cr> i took the medications for the prescribed time and symptoms improved, however, I still have some symptoms which are being treated through physical therapy since the accident was only in December     Ground: Medium Predicted: High , Probability 0.613825


```

And the first prediction sample from `test` now looks like:

```
        text    sentiment
0       Benefits: <cr> The antibiotic may have destroyed bacteria causing my sinus infection.  But it may also have been caused by a virus, so its hard to say. <cr> Side effects: <cr> Some back pain, some nauseau. <cr> Overall comments: <cr> Took the antibiotics for 14 days. Sinus infection was gone after the 6th day.  Low , Probability 0.370522
```

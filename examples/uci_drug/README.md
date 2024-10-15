### Fine-tuning for classification: End-to-end example
1. Download data from [Drug Review Dataset (Drugs.com) Data Set](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29) to `data` folder and extract. Pay attention to their terms:
   1. only use the data for research purposes
   2. don't use the data for any commerical purposes
   3. don't distribute the data to anyone else
   4. cite us

2. Run ```python -m cnlpt.data.transform_uci_drug <raw dir> <processed dir>``` to preprocess the data from the extract directory into a new directory. This will create {train,dev,test}.tsv in the processed directory specified, where the sentiment ratings have been collapsed into 3 categories.

3. Fine-tune with something like: 
```
python -m cnlpt.train_system \
	    --task_name sentiment \
	    --data_dir /lab-share/CHIP-Savova-e2/Public/mmtl-data/uci-drug/cnlpt/ \
	    --encoder_name roberta-base \
	    --do_train \
        --do_eval \
        --cache cache/ \
        --output_dir uci_error_sample_for_documentation/ \
        --overwrite_output_dir \
        --evals_per_epoch 5 \
	    --num_train_epochs 1 \
	    --learning_rate 1e-5 \
	    --report_to none \
        --save_strategy no
```

On our hardware, that command results in eval performance like the following:
```'eval_sentiment': {'acc': 0.8115933044017359, 'f1': [0.8981458951773809, 0.8000984130889407, 0.34115019542155217], 'acc_and_f1': [0.8548695997895583, 0.8058458587453383, 0.5763717499116441], 'recall': [0.9443307408923455, 0.8237082066869301, 0.25352697095435683], 'precision': [0.8562679781015125, 0.7778043530255919, 0.5213310580204779]}```

For a demo of how to run the system in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IVT53DBwFxLKftpIn5iKtF0g4xb9yuxm?usp=sharing)

#### Error Analysis / Human Readable Predictions for Classification

If you run the above command with the `--error_analysis` flag, you can obtain the `dev` instances for which the model made an erroneous 
prediction, organized by their original index in `dev` split, in the `eval_predictions...tsv` file in the `--output_dir` argument.  
For us the first line of this file is:
```
	text	sentiment
9	I'm 16 and  I have been on Loestrin 24 for about a week and half. The day I got them (and started taking them) was a day after I stopped my period and two days ago I started my period it is like a normal. I don't think I have gained weight due to me being sick and therefore don't eat as much as I normally do but I did not lose weight like I normally do when I'm sick. I have been getting cramps which I don't normally get except the first one or two days of my period. I have been really depressed and I'm not a depressed person. I mean I was crying over the stupidest things like my mom not cooking dinner when I wasn't even hungry. I'm going to talk to my doctor tomorrow.	Ground: Low Predicted: Medium
```
The `text` column contains the text of the erroneous instances and the following columns are the tasks provided to the model, in this case, just `sentiment`.  `Ground: Low Predicted: Medium` indicates that the provided ground truth label for the instance sentiment is `Low` but the model predicted `Medium`.  
Similarly if you run the above command with `--do_predict` you can obtain human readable predictions for the `test` split, in the `test_predictions...tsv` file.  For us the first line of this file is:
```
        text    sentiment
0       I've tried a few antidepressants over the years (citalopram, fluoxetine, amitriptyline), but none of those helped with my depression, insomnia &amp; anxiety. My doctor suggested and changed me onto 45mg mirtazapine and this medicine has saved my life. Thankfully I have had no side effects especially the most common - weight gain, I've actually lost alot of weight. I still have suicidal thoughts but mirtazapine has saved me.     High
```
##### Prediction Probability Outputs for Classification

(Currently only supported for classification tasks), if you run the above command with the `--output_prob` flag, you can see the model's softmax-obtained probability for the predicted classification label.  The first error analysis sample from `dev` would now looks like: 
```
	text	sentiment
9	I'm 16 and  I have been on Loestrin 24 for about a week and half. The day I got them (and started taking them) was a day after I stopped my period and two days ago I started my period it is like a normal. I don't think I have gained weight due to me being sick and therefore don't eat as much as I normally do but I did not lose weight like I normally do when I'm sick. I have been getting cramps which I don't normally get except the first one or two days of my period. I have been really depressed and I'm not a depressed person. I mean I was crying over the stupidest things like my mom not cooking dinner when I wasn't even hungry. I'm going to talk to my doctor tomorrow.	Ground: Low Predicted: Medium , Probability 0.728066
```
And the first prediction sample from `test` now looks like:
```
        text    sentiment
0       I've tried a few antidepressants over the years (citalopram, fluoxetine, amitriptyline), but none of those helped with my depression, insomnia &amp; anxiety. My doctor suggested and changed me onto 45mg mirtazapine and this medicine has saved my life. Thankfully I have had no side effects especially the most common - weight gain, I've actually lost alot of weight. I still have suicidal thoughts but mirtazapine has saved me.     High , Probability 0.995064
```

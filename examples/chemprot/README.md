### Fine-tuning for tagging: End-to-end example
>
> Note: this example is not runnable on ARM and non-Linux platforms, and is only tested on Linux x86_64 platforms.

1. Download the data from the [ChemProt website](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) to `data` folder.  Note, the evaluation on the data will approximate since it is done in terms of the preprocessed data, *not* ChemProt's specified evaluation method.

2. Unzip the folder and each of the contents in the folder. Make sure that each folder starts with `chemprot_`. For example, train data folder should have a name of `chemprot_training`.

3. Run `uv pip install -r chemprot_requirements.txt` to get dependencies for the preprocessing tokenization and sentence splitting.

4. Preprocess the data with `python examples/chemprot/transform_prot.py data/ChemProt_Corpus/ data/scispacy_chemprot/ chemprot`.  Note, this only gives us {train,dev}.tsv in `data/scispacy_chemprot` since there are no gold labels for the test data.

5. Fine-tune with something like:

```bash
cnlpt train \
 --task_name chemical_ner gene_ner end_to_end \
 --data_dir data/scispacy_chemprot/ \
 --encoder_name allenai/scibert_scivocab_uncased  \
 --do_train \
 --do_eval \
 --cache_dir cache/ \
 --output_dir temp/  \
 --overwrite_output_dir \
 --num_train_epochs 50 \
 --learning_rate 2e-5 \
 --lr_scheduler_type constant \
 --report_to none \
 --save_strategy no \
 --gradient_accumulation_steps 1 \
 --eval_accumulation_steps 10 \
 --weight_decay 0.2
```

6. On our hardware, the above hyperparameters gives us for chemical NER:

```
chemical_ner = {'acc': 0.9890791744279493, 'token_f1': [0.8892689725072849, 0.0, 0.9942603670482167], 'f1': 0.8890155834283543, 'report': '\n              precision    recall  f1-score   support\n\n    CHEMICAL       0.89      0.89      0.89      7876\n\n   micro avg       0.89      0.89      0.89      7876\n   macro avg       0.89      0.89      0.89      7876\nweighted avg       0.89      0.89      0.89      7876\n'}
```

For gene NER (`GENE-Y` are normalizable gene names and `GENE-N` are non-normalizable):

```
gene_ner = {'acc': 0.9853163799923996, 'token_f1': [0.6889471384884229, 0.835886652648443, 0.41860465116279066, 0.17142857142857143, 0.9948818383600992], 'f1': 0.7862532583344766, 'report': '\n              precision    recall  f1-score   support\n\n      GENE-N       0.70      0.67      0.69      2355\n      GENE-Y       0.84      0.83      0.83      5013\n\n   micro avg       0.79      0.78      0.79      7368\n   macro avg       0.77      0.75      0.76      7368\nweighted avg       0.79      0.78      0.79      7368\n'}
```

And finally for end to end relation extraction:

```
end_to_end = {'f1': [0.49406688241639696, 0.5513974419706301, 0.44155844155844154, 0.43023255813953487, 0.5196261682242991, 0.9997951988736138], 'acc': 0.9995671624840267, 'recall': [0.4312617702448211, 0.5464788732394367, 0.5964912280701754, 0.4111111111111111, 0.610989010989011, 0.999788247834258], 'precision': [0.5782828282828283, 0.5564053537284895, 0.35051546391752575, 0.45121951219512196, 0.45203252032520325, 0.9998021500096244], 'report_dict': {' CPR_3': {'precision': 0.5782828282828283, 'recall': 0.4312617702448211, 'f1-score': 0.49406688241639696, 'support': 531}, ' CPR_4': {'precision': 0.5564053537284895, 'recall': 0.5464788732394367, 'f1-score': 0.5513974419706301, 'support': 1065}, ' CPR_5': {'precision': 0.35051546391752575, 'recall': 0.5964912280701754, 'f1-score': 0.44155844155844154, 'support': 114}, ' CPR_6': {'precision': 0.45121951219512196, 'recall': 0.4111111111111111, 'f1-score': 0.43023255813953487, 'support': 180}, ' CPR_9': {'precision': 0.45203252032520325, 'recall': 0.610989010989011, 'f1-score': 0.5196261682242991, 'support': 455}, 'None': {'precision': 0.9998021500096244, 'recall': 0.999788247834258, 'f1-score': 0.9997951988736138, 'support': 5034187}, 'accuracy': 0.9995671624840267, 'macro avg': {'precision': 0.5647096380764656, 'recall': 0.5993533735814689, 'f1-score': 0.572779448530486, 'support': 5036532}, 'weighted avg': {'precision': 0.9995801633159224, 'recall': 0.9995671624840267, 'f1-score': 0.999570694686655, 'support': 5036532}}, 'report_str': '              precision    recall  f1-score   support\n\n       CPR_3       0.58      0.43      0.49       531\n       CPR_4       0.56      0.55      0.55      1065\n       CPR_5       0.35      0.60      0.44       114\n       CPR_6       0.45      0.41      0.43       180\n       CPR_9       0.45      0.61      0.52       455\n        None       1.00      1.00      1.00   5034187\n\n    accuracy                           1.00   5036532\n   macro avg       0.56      0.60      0.57   5036532\nweighted avg       1.00      1.00      1.00   5036532\n'}
```

#### Error Analysis for Tagging and End to End Relation Extraction

If you run the above command with the `--error_analysis` flag,  you can obtain the `dev` instances for which the model made an erroneous
prediction, organized by their original index in `dev` split, in the `eval_predictions...tsv` file in the `--output_dir` argument.  

On our hardware using the hyperparameters above, the first line of this file is:

```
        text    chemical_ner    gene_ner        end_to_end
2       The mechanisms of these CNS effects of DM have been suggested to be associated with the low-affinity , noncompetitive , N-methyl-d-aspartate ( NMDA ) antagonism of DM and/or the high-affinity DM/sigma receptors .   Ground: chemical: " DM" , chemical: " DM" , chemical: " DM/sigma" Predicted:  Ground:  Predicted: n: " N-methyl-d-aspartate" , y: " DM"        Ground:  Predicted: ( 22, 26,  CPR_9 )
```

The number at the beginning of the line, 2, is the index of the instance in the `dev` split.  The `text` column contains the text of the erroneous instances and the following columns are the tasks provided to the model, in this case, `chemical_ner` and `gene_ner`, both tagging tasks, and `end_to_end`, an end to end relation extraction task.  For `chemical_ner`, the result `Ground: chemical: " DM" , chemical: " DM" , chemical: " DM/sigma" Predicted:` indicates that the provided ground truth label has the two spans of "DM" and the span of " DM/sigma" as as chemicals,
but that the empty `Predicted:` indicates that the model did not tag these spans as `chemical`. For `end_to_end`,
the result `Ground:  Predicted: ( 22, 26,  CPR_9 )`
indicates that there are no relations between any of the tokens in the provided ground truth labels but
that the model predicted a `CPR_9` relationship between tokens 22 and 26 ("NMDA" and the second "DM" mention)

#### Human Readable Predictions for Tagging and End to End Relation Extraction

Note, for error analysis on tagging and end to end relation extraction tasks, we only print the disagreements between the ground truth and predictions in the relevant task cells.  
We do not process the ChemProt `test` split for processing, but if we copy `dev.tsv` to `test.tsv` and run the above command with `--do_predict`,
we can obtain human readable predictions (without error analysis) for `dev`.  For example, with the above instance we obtain:

```
        text    chemical_ner    gene_ner        end_to_end
        ...
2       The mechanisms of these CNS effects of DM have been suggested to be associated with the low-affinity , noncompetitive , N-methyl-d-aspartate ( NMDA ) antagonism of DM and/or the high-affinity DM/sigma receptors .   chemical: " N-methyl-d-aspartate" , chemical: " NMDA"   n: " N-methyl-d-aspartate" , y: " DM"  ( 22, 26,  CPR_9 )
```

Note that the model did tag certain spans as chemicals, namely " N-methyl-d-aspartate" and "NMDA",
given these span tags were not mentioned in the error analysis output we can infer they are also in the ground truth labels.

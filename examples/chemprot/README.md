### Fine-tuning for tagging: End-to-end example
1. Download the data from the [ChemProt website](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/) to ``data`` folder.  Note, the evaluation on the data will approximate since it is done in terms of the preprocessed data, *not* ChemProt's specified evaluation method.

2. Unzip the folder and each of the contents in the folder. Make sure that each folder starts with `chemprot_`. For example, train data folder should have a name of `chemprot_training`.

3. Run ```pip install scispacy https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz``` to get dependencies for the preprocessing tokenization and sentence splitting.

4. Preprocess the data with ```python -m cnlpt.data.transform_prot data/ChemProt_Corpus/ data/scispacy_chemprot/ chemprot```.  Note, this only gives us {train,dev}.tsv in `data/scispacy_chemprot` since there are no gold labels for the test data.

5. Fine-tune with something like:

```python -m cnlpt.train_system --task_name chemical_ner gene_ner --data_dir data/scispacy_chemprot/ --encoder_name allenai/scibert_scivocab_uncased  --do_train --do_eval --cache cache/ --output_dir temp/ --overwrite_output_dir --num_train_epochs 1 --learning_rate 5e-5 --lr_scheduler_type constant --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --report_to none --save_strategy no --gradient_accumulation_steps 1 --eval_accumulation_steps 10 --weight_decay 0.2``` 

6. On our hardware, the above hyperparameters gives us:

```chemical_ner = {'acc': 0.989129012764844, 'token_f1': [0.8892991278884715, 0.0, 0.9942847410774819], 'f1': 0.8890444967852824, 'report': '\n              precision    recall  f1-score   support\n\n    CHEMICAL       0.89      0.89      0.89      7876\n\n   micro avg       0.89      0.89      0.89      7876\n   macro avg       0.89      0.89      0.89      7876\nweighted avg       0.89      0.89      0.89      7876\n'}```

For chemical NER and the following for gene NER (`GENE-Y` are normalizable gene names and `GENE-N` are non-normalizable):
```gene_ner = {'acc': 0.9835720382010852, 'token_f1': [0.5503875968992249, 0.8223125230882896, 0.0, 0.0, 0.9949847518170479], 'f1': 0.7477207783371888, 'report': '\n              precision    recall  f1-score   support\n\n      GENE-N       0.70      0.45      0.55      2355\n      GENE-Y       0.76      0.88      0.82      5013\n\n   micro avg       0.75      0.75      0.75      7368\n   macro avg       0.73      0.67      0.68      7368\nweighted avg       0.74      0.75      0.73      7368\n'}```
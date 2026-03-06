# Fine-tuning for tagging: End-to-end example

1. Preprocess the data with `uv run examples/chemprot/prepare_chemprot_dataset.py`

2. Fine-tune for NER with something like:

```bash
uv run cnlpt train \
 --model_type proj \
 --encoder allenai/scibert_scivocab_uncased \
 --data_dir ./dataset \
 --task chemical_ner --task gene_ner \
 --output_dir ./train_output \
 --overwrite_output_dir \
 --do_train --do_eval \
 --num_train_epochs 3 \
 --learning_rate 2e-5 \
 --lr_scheduler_type constant \
 --save_strategy best \
 --gradient_accumulation_steps 1 \
 --eval_accumulation_steps 10 \
 --weight_decay 0.2
```

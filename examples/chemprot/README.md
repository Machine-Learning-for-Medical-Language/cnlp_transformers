# Fine-tuning for tagging: End-to-end example

1. Preprocess the data with `uv run examples/chemprot/prepare_chemprot_dataset.py data/chemprot`

2. Fine-tune with something like:

```bash
cnlpt train \ 
 --task_name chemical_ner gene_ner \
 --data_dir data/chemprot \
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

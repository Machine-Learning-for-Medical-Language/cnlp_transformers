#!/bin/bash

###########################
# Meant to do regression tests on all the different tasks that we have previously been able to handle.
# Use overwrite_cache to make sure the data processing is being tested. Use overwrite_output_dir and never put anything real in those output directories.

# Classification:

## sharp seed tasks:
for task in Negation Uncertainty Subject Historical; do
    python -m cnlpt.train_system --task ${task} --data_dir ~/mnt/r/DeepLearning/mmtl/assertion/seed/ --output_dir outputs/${task}_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache
done

## thyme colon event attribute tasks:
for task in dtr Negation Conmod alinkcat alinkb; do
    echo "************ Testing ${task} thyme colon (classification) *************"
    python -m cnlpt.train_system --task ${task} --data_dir ~/mnt/r/DeepLearning/mmtl/event-atts/colon/ --output_dir outputs/${task}_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_colon_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache
done

## Multi-task classification:
python -m cnlpt.train_system --task Negation Uncertainty --data_dir ~/mnt/r/DeepLearning/mmtl/assertion/seed/ --output_dir outputs/polarity+uncertainty_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name polarity+uncertainty_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache

# Tagging 

## thyme colon tasks:
for task in event timex; do
####
echo "********* Testing ${task} thyme colon (tagging) **************"
    python -m cnlpt.train_system --task ${task} --data_dir ~/mnt/r/DeepLearning/mmtl/thyme/colon/ --output_dir outputs/${task}_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache
done


## dphe-drug?

## joint tagging:
python -m cnlpt.train_system --task event timex --data_dir ~/mnt/r/DeepLearning/mmtl/thyme/colon/ --output_dir outputs/thyme_tagging_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name thyme_tagging_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache

# Relations
python -m cnlpt.train_system --task tlinkx --data_dir ~/mnt/r/DeepLearning/mmtl/thyme/colon/ --output_dir outputs/tlinkx_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name tlinkx_colon_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache

# Relations + Timexes + Events joint training:
# FIXME - not working
#python -m cnlpt.train_system --task event timex tlinkx --data_dir ~/mnt/r/DeepLearning/mmtl/thyme/colon/ --output_dir outputs/thyme_e2e_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name thyme_e2e_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache

# hier model
## i2b2 2008

## Mimic los
## FIXME - data file has output_mode mtl but that is being deprecated in the code.
python -m cnlpt.train_system --data_dir ~/mnt/r/DeepLearning/mmtl/mimic-los-short/ --output_dir outputs/mimic_los_regression_test --do_train --task_name mimic-los  --model hier --chunk_len 100 --num_chunks 10 --max_seq_length 1000 --per_device_train_batch_size 2 --save_steps -1 --save_total_limit 0 --evals_per_epoch 100 --report_to none --overwrite_cache --overwrite_output

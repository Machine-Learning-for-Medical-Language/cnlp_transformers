#!/bin/bash

data_dir='/home/ch150151/mmtl-data'
output_dir='/temp_work/ch150151/outputs/'

###########################
# Meant to do regression tests on all the different tasks that we have previously been able to handle.
# Use overwrite_cache to make sure the data processing is being tested. Use overwrite_output_dir and never put anything real in those output directories.

# Classification:

## sharp seed tasks:
for task in Negation Uncertainty Subject Historical; do
    cnlpt train --task_name ${task} --data_dir ${data_dir}/assertion/seed --output_dir ${output_dir}/${task}_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --eval_strategy epoch || break
    rm ${output_dir}/${task}_regression_test/pytorch_model.bin
done

## thyme colon event attribute tasks:
for task in dtr Negation Conmod alinkcat alinkb; do
    echo "************ Testing ${task} thyme colon (classification) *************"
    cnlpt train --task_name ${task} --data_dir ${data_dir}/thyme-event-atts/colon/ --output_dir ${output_dir}/${task}_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_colon_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0 || break
    rm ${output_dir}/${task}_colon_regression_test/pytorch_model.bin
done

## Multi-task classification: multiple tasks, one dataset
cnlpt train --task_name Negation Uncertainty --data_dir ${data_dir}/assertion/seed --output_dir ${output_dir}/polarity+uncertainty_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name polarity+uncertainty_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0

## Multi-dataset classification - one task, multiple datasets
cnlpt train --task_name TermExists --data_dir ${data_dir}/assertion/seed ${data_dir}/assertion/strat/ --output_dir ${output_dir}/term_exists_seed+strat_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --run_name termexists_seed+strat_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/term_exists_seed+strat_regression_test/pytorch_model.bin

## multi-dataset clasification - multiple tasks, multiple datasets:
cnlpt train --task_name Negation dtr --data_dir ${data_dir}/assertion/seed ${data_dir}/thyme-event-atts/colon/  --output_dir ${output_dir}/negation+dtr_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --run_name negation+dtr_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/negation+dtr_regression_test/pytorch_model.bin

## multi-dataset multi-task type - multiple task types, multiple datasets:
cnlpt train --task_name Negation timex --data_dir ${data_dir}/assertion/seed ${data_dir}/thyme/colon/  --output_dir ${output_dir}/negation+timex_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --run_name negation+timex_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/negation+timex_regression_test/pytorch_model.bin

## seed + thyme, negation + tlinkx
cnlpt train --task_name Negation tlinkx --data_dir ${data_dir}/assertion/seed ${data_dir}/thyme/colon/  --output_dir ${output_dir}/negation+tlinkx_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --run_name negation+tlinkx_regression_test --per_device_train_batch_size 32 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/negation+tlinkx_regression_test/pytorch_model.bin

# Tagging 

## thyme colon tasks:
for task in event timex; do
####
echo "********* Testing ${task} thyme colon (tagging) **************"
    cnlpt train --task_name ${task} --data_dir ${data_dir}/thyme/colon/ --output_dir ${output_dir}/${task}_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name ${task}_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0 || break
    rm ${output_dir}/${task}_colon_regression_test/pytorch_model.bin
done


## dphe-drug?

## joint tagging:
cnlpt train --task_name event timex --data_dir ${data_dir}/thyme/colon/ --output_dir ${output_dir}/thyme_tagging_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name thyme_tagging_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/thyme_tagging_colon_regression_test/pytorch_model.bin

# Relations
cnlpt train --task_name tlinkx --data_dir ${data_dir}/thyme/colon/ --output_dir ${output_dir}/tlinkx_colon_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name tlinkx_colon_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/tlinkx_colon_regression_test/pytorch_model.bin

# Relations + Timexes + Events joint training:
cnlpt train --task_name event timex tlinkx --data_dir ${data_dir}/thyme/colon/ --output_dir ${output_dir}/thyme_e2e_regression_test --do_train --do_eval --num_train_epochs 1 --overwrite_output_dir --learning_rate 5e-5 --run_name thyme_e2e_regression_test --per_device_train_batch_size 8 --report_to none --overwrite_cache --save_steps -1 --save_total_limit 0
rm ${output_dir}/thyme_e2e_regression_test/pytorch_model.bin

# hier model
## i2b2 2008

## Mimic los
## FIXME - data file has output_mode mtl but that is being deprecated in the code.
cnlpt train --encoder microsoft/xtremedistil-l6-h256-uncased --data_dir ${data_dir}/mimic3-benchmark --output_dir ${output_dir}/mimic_los_regression_test --do_train --task_name "3-" "7-"  --model hier --chunk_len 100 --num_chunks 10 --max_seq_length 1000 --per_device_train_batch_size 2 --save_steps -1 --save_total_limit 0 --evals_per_epoch 100 --report_to none --overwrite_cache --overwrite_output
rm ${output_dir}/mimic_los_regression_test/pytorch_model.bin

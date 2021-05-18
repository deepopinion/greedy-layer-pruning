#!/bin/bash

source env/bin/activate

#
# Prune greedy
#
for task in "rte" "mrpc" "stsb" "cola" "sst2" "qnli" "qqp" "mnli"; do
for model in "bert-base-uncased" "roberta-base"; do
python3 prune.py --model_name_or_path=$model \
    --task_name=$task \
    --seed=41 \
    --max_seq_length=128 \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --output_dir=experiments/tmp/ \
    --logging_dir=experiments/tmp/ \
    --prune_n_layers=6 \
    --prune_method="greedy" \
    --overwrite_output_dir
done
done


#
# Prune optimal
#
for task in "sst2" "cola" "mnli" "mrpc" "qnli" "qqp" "rte" "stsb"; do
for model in "bert-base-uncased" "roberta-base"; do
python3 analyze.py --model_name_or_path=$model \
    --task_name=$task \
    --seed=41 \
    --max_seq_length=128 \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --output_dir=experiments/tmp/ \
    --logging_dir=experiments/tmp/ \
    --prune_n_layers=2 \
    --prune_method="optimal" \
    --overwrite_output_dir
done
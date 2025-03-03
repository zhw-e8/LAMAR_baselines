#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines


python \
evaluate.py \
--model_state_path=${work_dir}/UTR5TEPred/saving_model/RNAFM/bs4_lr1e-4_wr0.05_1epochs_5/checkpoint-2200/pytorch_model.bin \
--data_path=${work_dir}/UTR5TEPred/data/validation_set.csv \
--head_type=Linear
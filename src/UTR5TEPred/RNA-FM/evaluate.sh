#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/UTR5TEPred/RNA-FM/evaluate.py \
--model_state_path=UTR5TEPred/saving_model/RNAFM/bs4_lr1e-4_wr0.05_1epochs_5/checkpoint-2200/pytorch_model.bin \
--data_path=UTR5TEPred/data/validation_set.csv \
--head_type=Linear
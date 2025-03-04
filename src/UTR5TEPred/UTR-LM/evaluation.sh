#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/UTR5TEPred/UTR-LM/evaluation.py \
--model_state_path=UTR5TEPred/saving_model/UTR-LM/bs8_lr1e-4_wr0.05_1epochs_5/checkpoint-1100/pytorch_model.bin \
--data_path=UTR5TEPred/data/validation_set.csv
#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/IRESPred/RNA-FM/evaluation.py \
--model_state_path=IRESPred/saving_model/RNA-FM/bs4_lr1e-4_wr0.05_1epochs_4/checkpoint-400/pytorch_model.bin \
--data_path=IRESPred/data/testing_set.Pos1Fold.Train1Fold.4.csv \
--head_type=Linear
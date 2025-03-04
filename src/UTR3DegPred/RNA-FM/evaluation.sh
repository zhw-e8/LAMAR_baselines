#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/UTR3DegPred/RNA-FM/evaluation.py \
--model_state_path=UTR3DegPred/saving_model/RNA-FM/bs16_lr1e-4_wr0.05_10epochs_2/checkpoint-900/pytorch_model.bin \
--data_path=UTR3DegPred/data/validation_set.csv \
--head_type=Linear
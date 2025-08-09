#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/UTR5TEPred/RNAErnie/evaluation.py \
--model_state_path=UTR5TEPred/saving_model/RNAErnie/bs16_lr5e-5_wr0.05_16epochs_0/checkpoint-8830/model.safetensors \
--data_path=UTR5TEPred/data/validation_set.csv

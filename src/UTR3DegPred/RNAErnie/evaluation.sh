#!/bin/bash

# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}

python \
src/UTR3DegPred/RNAErnie/evaluation.py \
--model_state_path=UTR3DegPred/saving_model/RNAErnie/bs8_lr5e-5_wr0.05_2epochs_2/checkpoint-390/model.safetensors \
--data_path=UTR3DegPred/data/validation_set.csv

#!/bin/bash

echo 'Fine-tuning RNAErnie to predict translation efficiency'
echo "Batch size:" ${1}
echo "Total epochs:" ${2}
echo "Learning rate:" ${3}
echo "Sub-data ID:" ${4}

# Set batch size per GPU
if [ ${1} -gt 32 ]; then as=`expr ${1} / 32`; else as=1; fi
if [ ${1} -gt 32 ]; then bs=32; else bs=${1}; fi
# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}


CUDA_VISIBLE_DEVICES=${5} \
python \
src/UTR5TEPred/RNAErnie/finetune.py \
--tokenizer_path=tokenizer/RNAErnie \
--model_max_length=1500 \
--data_path=UTR5TEPred/data/te_RNAErnie_${4} \
--batch_size=${bs} \
--peak_lr=${3} \
--warmup_ratio=0.05 \
--total_epochs=${2} \
--grad_clipping_norm=1 \
--accum_steps=${as} \
--output_dir=UTR5TEPred/saving_model/RNAErnie/new/bs${1}_lr${3}_wr0.05_${2}epochs_${4} \
--logging_steps=100 \
--save_epochs=10

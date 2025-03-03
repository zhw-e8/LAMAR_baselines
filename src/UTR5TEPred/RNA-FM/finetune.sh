#!/bin/bash

echo "Fine-tuning RNA-FM to predict translation efficiency"
echo "Batch size:" ${1}
echo "Total epochs:" ${2}
echo "Learning rate:" ${3}
echo "Sub-data ID:" ${4}

# Set batch size per GPU
as=`expr ${1} / 2`
# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines


python \
finetune.py \
--tokenizer_path=${work_dir}/tokenizer/RNA-FM \
--model_max_length=1026 \
--hidden_dropout_prob=0 \
--data_path=${work_dir}/UTR5TEPred/data/te_RNA-FM_${4} \
--head_type=Linear \
--batch_size=2 \
--peak_lr=${3} \
--warmup_ratio=0.05 \
--total_epochs=${2} \
--grad_clipping_norm=1 \
--accum_steps=${as} \
--output_dir=${work_dir}/UTR5TEPred/saving_model/RNAFM/te_bs${1}_lr${3}_wr0.05_${2}epochs_${4} \
--logging_steps=100 \
--save_epochs=100

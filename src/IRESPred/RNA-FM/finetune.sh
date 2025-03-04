#!/bin/bash

echo 'Fine-tuning RNA-FM to predict IRES'
echo "Batch size:" ${1}
echo "Total epochs:" ${2}
echo "Learning rate:" ${3}
echo "Sub-data ID:" ${4}

# Set batch size per GPU
echo ${1}_${2}_${3}_${4}
as=`expr ${1} / 2`
# Set work directory
work_dir=/work/home/rnasys/zhouhanwen/github/LAMAR_baselines
cd ${work_dir}


python \
src/IRESPred/RNA-FM/finetune.py \
--tokenizer_path=tokenizer/RNA-FM \
--model_max_length=1500 \
--hidden_dropout_prob=0 \
--data_path=IRESPred/data/IRES_RNA-FM_${4} \
--head_type=Linear \
--batch_size=2 \
--peak_lr=${3} \
--warmup_ratio=0.05 \
--total_epochs=${2} \
--grad_clipping_norm=1 \
--accum_steps=${as} \
--output_dir=IRESPred/saving_model/RNA-FM/bs${1}_lr${3}_wr0.05_${2}epochs_${4} \
--logging_steps=100 \
--save_epochs=100

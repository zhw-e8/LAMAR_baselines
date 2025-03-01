#!/bin/bash

#SBATCH --job-name=TE
#SBATCH -p xahdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH -o slurm.%x.%j.out

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/gcc-7.3.1
module load compiler/dtk/22.10

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

# source /opt/rocm/set_env.sh
source /work/home/rnasys/zhouhanwen/nucTran/src/nucTran/torch_lib.sh
source /public/software/apps/DeepLearning/PyTorch_Lib/torch_env.sh
source /public/software/apps/DeepLearning/whl/pytorch_env.sh
source /work/home/rnasys/anaconda3/bin/activate RNA-FM
cat /work/home/rnasys/zhouhanwen/nucTran/UTR5TEPred/src/Timothy/RNAFM/finetune/finetune.sh


echo ${1}_${2}_${3}_${4}
as=`expr ${1} / 2`


python \
finetune.py \
--tokenizer_path=/work/home/rnasys/zhouhanwen/nucTran/tokenizer/rnafm \
--model_max_length=1026 \
--hidden_dropout_prob=0 \
--data_path=/work/home/rnasys/zhouhanwen/nucTran/UTR5TEPred/data/Timothy/training/te_rnafm_${4} \
--head_type=Linear \
--batch_size=2 \
--peak_lr=${3} \
--warmup_ratio=0.05 \
--total_epochs=${2} \
--grad_clipping_norm=1 \
--accum_steps=${as} \
--output_dir=/work/home/rnasys/zhouhanwen/nucTran/UTR5TEPred/saving_model/Timothy/RNAFM/te_bs${1}_lr${3}_wr0.05_${2}epochs_${4} \
--logging_steps=100 \
--save_epochs=100

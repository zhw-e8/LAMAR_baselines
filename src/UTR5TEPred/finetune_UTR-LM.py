from sequence_classification_patch import Config, UTRLMForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_from_disk
import torch
import os
import numpy as np
import pandas as pd
import argparse


def compute_metrics(p):
    """
    labels: true labels
    predictions: predict labels
    """
    predictions, labels = p
    predictions = predictions[0].squeeze()
    mse = np.mean((predictions - labels) ** 2)
    df = pd.DataFrame({'pred': predictions, 'label': labels})
    corr_coef_pearson = df.corr(method='pearson').iloc[0, 1]
    corr_coef_spearman = df.corr(method='spearman').iloc[0, 1]
    
    return {
        "mse": mse,
        "corr_coef_pearson": corr_coef_pearson, 
        "corr_coef_spearman": corr_coef_spearman
    }


def main(
        tokenizer_path, 
        model_max_length, 
        model_name, 
        hidden_dropout_prob, 
        data_path, 
        head_type, 
        freeze, 
        optimizer_name, 
        batch_size, 
        peak_lr, 
        warmup_ratio, 
        total_epochs, 
        grad_clipping_norm, 
        accum_steps, 
        output_dir, 
        logging_steps, 
        save_epochs, 
        fp16
    ):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length, padding_side='left')
    # Config
    hyperparams = Config(model_name, hidden_size=128, num_labels=1, hidden_dropout_prob=hidden_dropout_prob, initializer_range=0.02)
    # Training data
    data = load_from_disk(data_path)
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True
    )
    # Model
    model = UTRLMForSequenceClassification(hyperparams, head_type='Linear', freeze=False)
    if model_name == 'ESM2':
        pretrain_state_path = '/work/home/rnasys/zhouhanwen/nucTran/src/UTRLM/Pretrained/ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl'
    elif model_name == 'ESM2_SISS':
        pretrain_state_path = '/work/home/rnasys/zhouhanwen/nucTran/src/UTRLM/Pretrained/ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl'
    weights = torch.load(pretrain_state_path, map_location='cpu')
    weights_dict = {}
    for k, v in weights.items():
        if hyperparams.model == 'ESM2':
            new_k = 'utrlm.' + k
        elif hyperparams.model == 'ESM2_SISS':
            new_k = k.replace('module', 'utrlm') if 'module' in k else k
        weights_dict[new_k] = v
    model.load_state_dict(weights_dict, strict=False)
    if optimizer_name == 'AdamW':
        # Training arguments
        train_args = TrainingArguments(
            disable_tqdm=True, 
            save_total_limit=1, 
            dataloader_drop_last=True, 
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=1, 
            learning_rate=peak_lr, 
            weight_decay=0.01, 
            adam_beta1=0.9, 
            adam_beta2=0.98, 
            adam_epsilon=1e-8, 
            warmup_ratio=warmup_ratio, 
            num_train_epochs=total_epochs, 
            max_grad_norm=grad_clipping_norm, 
            gradient_accumulation_steps=accum_steps, 
            output_dir=output_dir, 
            evaluation_strategy="steps",
            eval_steps=logging_steps,
            save_strategy='steps', 
            save_steps=save_epochs, 
            logging_strategy = 'steps', 
            logging_steps=logging_steps, 
            fp16=fp16, 
            save_safetensors=False
        )
        # Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=data['train'], 
            eval_dataset=data['test'], 
            compute_metrics=compute_metrics, 
            data_collator=data_collator, 
            tokenizer=tokenizer
        )
    elif optimizer_name == 'SGD':
        # Training arguments
        train_args = TrainingArguments(
            disable_tqdm=True, 
            save_total_limit=1, 
            dataloader_drop_last=True, 
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=1, 
            learning_rate=peak_lr, 
            num_train_epochs=total_epochs, 
            max_grad_norm=grad_clipping_norm, 
            gradient_accumulation_steps=accum_steps, 
            output_dir=output_dir, 
            evaluation_strategy="steps",
            eval_steps=logging_steps,
            save_strategy='steps', 
            save_steps=save_epochs, 
            logging_strategy = 'steps', 
            logging_steps=logging_steps, 
            fp16=fp16, 
            save_safetensors=False
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=train_args.learning_rate, momentum=0.9, weight_decay=1e-4)
        # Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=data['train'], 
            eval_dataset=data['test'], 
            optimizers=(optimizer, None), 
            compute_metrics=compute_metrics, 
            data_collator=data_collator, 
            tokenizer=tokenizer
        )
    # Training
    trainer.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5 prime UTR TE prediction')
    parser.add_argument('--tokenizer_path', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', type=int, help='Model input size')
    parser.add_argument('--model_name', type=str, help='Name of pretrained model')
    parser.add_argument('--hidden_dropout_prob', type=float, help='Hidden dropout probability')
    parser.add_argument('--data_path', type=str, help='Path of the data for inference')
    parser.add_argument('--head_type', type=str, help='Type of classification head')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained weights')
    parser.add_argument('--optimizer_name', type=str, help='Name of optimizer')
    parser.add_argument('--batch_size', type=int, help='Input batch size on each device')
    parser.add_argument('--peak_lr', type=float, help='Peak learning rate')
    parser.add_argument('--warmup_ratio', type=float, help='Warm up ratio')
    parser.add_argument('--total_epochs', type=int, help='Epochs for training')
    parser.add_argument('--grad_clipping_norm', type=float, help='Gradient clipping norm')
    parser.add_argument('--accum_steps', type=int, help='Accumulation steps')
    parser.add_argument('--output_dir', type=str, help='Output dir')
    parser.add_argument('--logging_steps', type=int, help='Steps for logging')
    parser.add_argument('--save_epochs', type=int, help='Steps for saving')
    parser.add_argument('--fp16', action='store_true', help='FP16 training')
    args = parser.parse_args()

    main(
        args.tokenizer_path, 
        args.model_max_length, 
        args.model_name, 
        args.hidden_dropout_prob, 
        args.data_path,  
        args.head_type, 
        args.freeze, 
        args.optimizer_name, 
        args.batch_size, 
        args.peak_lr, 
        args.warmup_ratio, 
        args.total_epochs, 
        args.grad_clipping_norm, 
        args.accum_steps, 
        args.output_dir, 
        args.logging_steps, 
        args.save_epochs, 
        args.fp16
    )

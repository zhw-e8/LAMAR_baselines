import sys
sys.path.append('RNAErnie/')
from tokenization_rnaernie import RNAErnieTokenizer
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_from_disk
import os
import numpy as np
import pandas as pd
from safetensors.torch import load_file, load_model
import argparse


def compute_metrics(p):
    """
    labels: true labels
    predictions: predict labels
    """
    predictions, labels = p
    predictions = predictions.squeeze()
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
        data_path, 
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
    tokenizer = RNAErnieTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    config = AutoConfig.from_pretrained(
        'RNAErnie/config.json', vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, num_labels=1
    )
    # Training data
    data = load_from_disk(data_path)
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True
    )
    # Model
    model = AutoModelForSequenceClassification.from_config(config)
    load_model(model, filename='RNAErnie/model.safetensors', strict=False)
    # Training arguments
    train_args = TrainingArguments(
        disable_tqdm=True, 
        save_total_limit=1, 
        dataloader_drop_last=True, 
        per_device_train_batch_size=batch_size, 
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
        fp16=fp16
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
    # Training
    trainer.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3 prime UTR degradation prediction')
    parser.add_argument('--tokenizer_path', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', type=int, help='Model input size')
    parser.add_argument('--data_path', type=str, help='Path of the data for inference')
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
        args.data_path,  
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


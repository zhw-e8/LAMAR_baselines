import sys
sys.path.append('RNA-FM/')
from sequence_classification_patch import Config, RnafmForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_from_disk
import os
import numpy as np
import pandas as pd
import evaluate
from sklearn.metrics import precision_recall_curve, auc
import argparse


def compute_metrics(p):
    """
    labels: true labels
    predictions: predict labels
    pred_probs: predict probabilities
    """
    # metrics
    accuracy = evaluate.load("metrics/accuracy")
    precision = evaluate.load("metrics/precision")
    recall = evaluate.load("metrics/recall")
    f1 = evaluate.load("metrics/f1")
    roc_auc = evaluate.load("metrics/roc_auc")
    
    predictions, labels = p
    predictions = predictions[0]
    pred_probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    predictions = np.argmax(predictions, axis=1).flatten()
    labels = np.array(labels).flatten()
    
    accuracy_v = accuracy.compute(references=labels, predictions=predictions)
    precision_v = precision.compute(references=labels, predictions=predictions, zero_division=0)
    recall_v = recall.compute(references=labels, predictions=predictions)
    f1_v = f1.compute(references=labels, predictions=predictions)
    roc_auc_v = roc_auc.compute(references=labels, prediction_scores=pred_probs[:, 1])
    precision_prauc, recall_prauc, threshold_prauc = precision_recall_curve(labels, pred_probs[:, 1])
    pr_auc_v = auc(recall_prauc, precision_prauc) 

    return {
        "accuracy": accuracy_v['accuracy'],
        "precision": precision_v['precision'],
        "recall": recall_v['recall'],
        "f1": f1_v['f1'], 
        "roc_auc": roc_auc_v['roc_auc'], 
        "pr_auc": pr_auc_v
    }


def main(
        tokenizer_path, 
        model_max_length, 
        hidden_dropout_prob, 
        data_path, 
        head_type, 
        freeze, 
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    hyperparams = Config(hidden_size=640, num_labels=2, hidden_dropout_prob=hidden_dropout_prob)
    # Training data
    data = load_from_disk(data_path)
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True
    )
    # Model
    pretrained_state_path = 'RNA-FM/RNA-FM_pretrained.pth'
    model = RnafmForSequenceClassification(pretrained_weights_location=pretrained_state_path, hyperparams=hyperparams, head_type=head_type, freeze=freeze)    
    # Training arguments
    train_args = TrainingArguments(
        disable_tqdm=True, 
        save_total_limit=1, 
        dataloader_drop_last=True, 
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=2, 
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
    # Training
    trainer.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRES prediction')
    parser.add_argument('--tokenizer_path', type=str, help='Directory of tokenizer')
    parser.add_argument('--model_max_length', type=int, help='Model input size')
    parser.add_argument('--hidden_dropout_prob', type=float, help='Hidden dropout probability')
    parser.add_argument('--data_path', type=str, help='Path of the data for inference')
    parser.add_argument('--head_type', type=str, help='Type of classification head')
    parser.add_argument('--freeze', action='store_true', help='Freeze pretrained weights')
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
        args.hidden_dropout_prob, 
        args.data_path,  
        args.head_type, 
        args.freeze, 
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

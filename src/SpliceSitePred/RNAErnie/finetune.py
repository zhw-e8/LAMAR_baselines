import sys
sys.path.append('/work/home/rnasys/zhouhanwen/github/LAMAR_baselines/RNAErnie')
from tokenization_rnaernie import RNAErnieTokenizer
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForTokenClassification
from datasets import load_dataset, load_from_disk
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.special import softmax
import os
import torch
import numpy as np
from safetensors.torch import load_file, load_model
import evaluate
import pandas as pd
import argparse


def compute_binary_pr_auc(reference, predict_logits):
    precision, recall, _ = precision_recall_curve(reference, predict_logits)
    return auc(recall, precision)


def compute_ovr_pr_auc(reference, predict_logits, average=None, ignore_idx=[]):
    n_classes = predict_logits.shape[1]
    pr_aucs = []
    for class_idx in range(n_classes):
        if class_idx not in ignore_idx:
            pr_auc = compute_binary_pr_auc((reference == class_idx).astype(int), predict_logits[:, class_idx])
            pr_aucs.append(pr_auc)
    if average == "macro":
        return np.mean(pr_aucs)
    elif average == "weighted":
        class_counts = np.bincount(reference)
        weighted_pr_aucs = np.array(pr_aucs) * class_counts / len(reference)
        return np.sum(weighted_pr_aucs)
    else:
        return pr_aucs


def compute_ovo_pr_auc(reference, predict_logits, average=None):
    # OvO is not directly supported by precision_recall_curve
    raise NotImplementedError("OvO PR AUC computation is not implemented yet.")


def pr_auc_score(reference, predict_logits, multi_class=None, average=None):
    if multi_class == "ovr":
        pr_auc = compute_ovr_pr_auc(reference, predict_logits, average=average)
    elif multi_class == "ovo":
        pr_auc = compute_ovo_pr_auc(reference, predict_logits, average=average)
    else:
        pr_auc = compute_binary_pr_auc(reference, predict_logits)
    return pr_auc


def compute_metrics(p):
    ignore_label = -100
    logits, labels = p
    softpred = softmax(logits, axis=2)
    pred_label = np.argmax(softpred, axis=2).astype(np.int8)
    logits = softpred.reshape((softpred.shape[0] * softpred.shape[1], -1))
    table = pd.DataFrame(logits)
    table["pred"] = np.array(pred_label).flatten()
    table["true"] = np.array(labels).flatten()
    table = table[table["true"] != ignore_label]
    # print("finish flatten")
    result = {}
    counts = table.true.value_counts().to_dict()
    result["topk"] = {
        "topk": {k: sum((table.sort_values(by=k, ascending=False)[:v]).true == k) / v for k, v in counts.items()}
    }
    scores = table.loc[:, table.columns[~table.columns.isin(["pred", "true"])]].values
    result["pr_auc"] = list(
        pr_auc_score(
            table["true"],
            scores,
            multi_class="ovr",
            average=None
        )
    )
    return result


def main(batch_size, peak_lr, total_epochs, accum_steps, save_dir):
    os.chdir('/work/home/rnasys/zhouhanwen/github/LAMAR_baselines/')
    tokenizer_path = 'tokenizer/RNAErnie/'
    model_name = 'RNAErnie/config.json'
    model_max_length = 1026
    nlabels = 3
    data_path = 'SpliceSitePred/data/RNAErnie/ss_single_nucleotide/'
    pretrain_state_path = 'RNAErnie/model.safetensors'
    # pretrain_state_path = None
    warmup_ratio = 0.05
    grad_clipping_norm = 1
    accum_steps = accum_steps
    output_dir = 'SpliceSitePred/saving_model/RNAErnie/{}'.format(save_dir)
    save_epochs = 500
    logging_steps = 500
    fp16 = False
    # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    tokenizer = RNAErnieTokenizer.from_pretrained('tokenizer/RNAErnie/', model_max_length=model_max_length)
    # Config
    config = AutoConfig.from_pretrained(
        model_name, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, num_labels=nlabels, 
        classifier_dropout=0
    )
    # Training data
    data = load_from_disk(data_path)
    # Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True
    )
    # Model
    model = AutoModelForTokenClassification.from_config(config)
    print("Loading parameters of pretraining model: {}".format(pretrain_state_path))
    load_model(model, filename=pretrain_state_path, strict=False)
    # Training arguments
    train_args = TrainingArguments(
        disable_tqdm=True, 
        save_total_limit=20, 
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
        report_to="none"
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
    parser = argparse.ArgumentParser(description='Splice site prediction')
    parser.add_argument('--batch_size', type=int, help='Input batch size on each device')
    parser.add_argument('--peak_lr', type=float, help='Peak learning rate')
    parser.add_argument('--total_epochs', type=int, help='Epochs for training')
    parser.add_argument('--accum_steps', type=int, help='Accumulative steps')
    parser.add_argument('--save_dir', type=str, help='Directory')
    args = parser.parse_args()

    main(
        args.batch_size, 
        args.peak_lr, 
        args.total_epochs, 
        args.accum_steps,
        args.save_dir
    )
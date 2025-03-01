from sequence_classification_patch import Config, RnafmForSequenceClassification
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import os
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc


def compute_pr_auc(true_ids, probs):
    """
    Compute the PR-AUC score
    Input:
        true_ids
        logits, raw model logits
    Return:
        pr_auc, float
    """
    
    precision, recall, threshold = precision_recall_curve(true_ids, probs)
    pr_auc = auc(recall, precision)
    
    return pr_auc


def main(model_state_path, data_path, head_type, freeze):
    os.chdir('/work/home/rnasys/zhouhanwen/nucTran')
    # Tokenizer
    tokenizer_path = 'tokenizer/rnafm/'
    model_max_length = 1500
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    hidden_size = 640
    nlabels = 2
    hidden_dropout_prob = 0
    hyperparams = Config(hidden_size, nlabels, hidden_dropout_prob)
    # Inference data
    seq_df = pd.read_csv(data_path)
    seqs = seq_df['seq'].values.tolist()
    true_labels = seq_df['label'].values.tolist()
    # Model
    device = torch.device('cuda:0')
    pretrained_state_path = '/work/home/rnasys/zhouhanwen/nucTran/src/RNAFM/RNA-FM-main/src/RNAFM/RNA-FM_pretrained.pth'
    model = RnafmForSequenceClassification(pretrained_weights_location=pretrained_state_path, hyperparams=hyperparams, head_type=head_type, freeze=freeze)    
    model = model.to(device)
    if model_state_path.endswith('.safetensors'):
        load_model(model, filename=model_state_path, strict=True)
    elif model_state_path.endswith('.bin'):
        model.load_state_dict(torch.load(model_state_path), strict=True)
    
    softmax = nn.Softmax(dim=1)
    predict_labels, predict_probs = [], []
    model.eval()
    with torch.no_grad():
        for seq in tqdm.tqdm(seqs):
            batch = tokenizer(seq, return_tensors='pt')
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            model_output = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                position_ids = None,
                head_mask = None,
                inputs_embeds = None,
                labels = None,
                output_attentions = None,
                output_hidden_states = None,
                return_dict = None
            )
            batch_logits = model_output.logits
            predict_probs.extend(softmax(batch_logits)[:, 1].tolist())
            predict_labels.extend(batch_logits.argmax(dim=1).tolist())

    result_df = pd.DataFrame({'predict': predict_labels, 'true': true_labels})
    precision = precision_score(true_labels, predict_labels)
    recall = recall_score(true_labels, predict_labels)
    f1 = f1_score(true_labels, predict_labels)
    roc_auc = roc_auc_score(true_labels, predict_probs)
    pr_auc = compute_pr_auc(true_labels, predict_probs)
    
    return precision, recall, f1, roc_auc, pr_auc
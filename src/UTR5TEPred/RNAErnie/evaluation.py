import sys
sys.path.append('RNAErnie/')
from tokenization_rnaernie import RNAErnieTokenizer
import torch
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForSequenceClassification
import os
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import r2_score
from safetensors.torch import load_model
import argparse


def main(model_state_path, data_path):
    # Tokenizer
    tokenizer_path = 'tokenizer/RNAErnie/'
    model_max_length = 1500
    tokenizer = RNAErnieTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    config = AutoConfig.from_pretrained(
        'RNAErnie/config.json', vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, num_labels=1, 
    )
    # Inference data
    seq_df = pd.read_csv(data_path)
    seqs = seq_df['seq'].values.tolist()
    true_labels = seq_df['label'].values.tolist()
    # Model
    device = torch.device('cuda:0')
    model = AutoModelForSequenceClassification.from_config(config)
    model = model.to(device)
    if model_state_path.endswith('.safetensors'):
        load_model(model, filename=model_state_path, strict=True)
    elif model_state_path.endswith('.bin'):
        model.load_state_dict(torch.load(model_state_path), strict=True)
    
    predict_labels = []
    model.eval()
    with torch.no_grad():
        for seq in tqdm.tqdm(seqs):
            batch = tokenizer(seq, return_tensors='pt', padding=True)
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
            predict_labels.extend(batch_logits.tolist()[0])
    result_df = pd.DataFrame({'predict': predict_labels, 'true': true_labels})
    mse = np.mean((np.array(predict_labels) - np.array(true_labels)) ** 2)
    pearson_corr_coef = result_df.corr(method='pearson').iloc[0, 1]
    spearman_corr_coef = result_df.corr(method='spearman').iloc[0, 1]
    r2 = r2_score(np.array(true_labels), np.array(predict_labels))
    print('MSE: {}, Pearson Corr: {}, Spearman Corr: {}, R2: {}'.format(mse, pearson_corr_coef, spearman_corr_coef, r2))
    return mse, pearson_corr_coef, spearman_corr_coef, r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5 prime UTR translation prediction')
    parser.add_argument('--model_state_path', type=str, help='Path of fine-tuned model')
    parser.add_argument('--data_path', type=str, help='Path of validation dataset')
    args = parser.parse_args()

    main(
        args.model_state_path, 
        args.data_path
    )

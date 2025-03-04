import sys
sys.path.append('UTR-LM/')
from sequence_classification_patch import Config, UTRLMForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import os
import numpy as np
import pandas as pd
import tqdm
import argparse
    
    
def main(model_state_path, data_path):
    # Tokenizer
    tokenizer_path = 'tokenizer/UTR-LM/'
    model_max_length = 1026
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length, padding_side='left')
    # Config
    model_type = 'ESM2'
    hidden_size = 128
    nlabels = 1
    hidden_dropout_prob = 0
    initializer_range = 0.02
    hyperparams = Config(model_type, hidden_size, nlabels, hidden_dropout_prob, initializer_range)
    # Inference data
    seq_df = pd.read_csv(data_path)
    seqs = seq_df['seq'].values.tolist()
    true_labels = seq_df['label'].values.tolist()
    # Model
    device = torch.device('cuda:0')
    model = UTRLMForSequenceClassification(
        hyperparams=hyperparams, 
        head_type='Linear', 
        freeze=False
    )
    model = model.to(device)
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
    print('MSE: {}, Pearson Corr: {}, Spearman Corr: {}'.format(mse, pearson_corr_coef, spearman_corr_coef))
    return mse, pearson_corr_coef, spearman_corr_coef


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5 prime UTR TE prediction')
    parser.add_argument('--model_state_path', type=str, help='Path of fine-tuned model')
    parser.add_argument('--data_path', type=str, help='Path of validation dataset')
    args = parser.parse_args()

    main(
        args.model_state_path, 
        args.data_path
    )
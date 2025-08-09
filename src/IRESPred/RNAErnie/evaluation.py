from tokenization_rnaernie import RNAErnieTokenizer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file, load_model
from torch import nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import argparse


def main(model_state_path, data_path):
    # Tokenizer
    tokenizer_path = 'tokenizer/RNAErnie/'
    model_max_length = 1500
    tokenizer = RNAErnieTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length)
    # Config
    config = AutoConfig.from_pretrained(
        'RNAErnie/config.json', vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, mask_token_id=tokenizer.mask_token_id, num_labels=2, 
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

    precision_v = precision_score(true_labels, predict_labels)
    recall_v = recall_score(true_labels, predict_labels)
    f1_v = f1_score(true_labels, predict_labels)
    auc_v = roc_auc_score(true_labels, predict_probs)
    print('Precision: {}, Recall: {}, F1: {}, AUC: {}'.format(precision_v, recall_v, f1_v, auc_v))
    return precision_v, recall_v, f1_v, auc_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRES prediction')
    parser.add_argument('--model_state_path', type=str, help='Path of fine-tuned model')
    parser.add_argument('--data_path', type=str, help='Path of validation dataset')
    args = parser.parse_args()

    main(
        args.model_state_path, 
        args.data_path
    )

from sequence_classification_patch import Config, RnafmForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
import os
import numpy as np
import pandas as pd
import tqdm


class MyDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        
        self.size = len(df)
        self.data = []
        for i in range(self.size):
            self.data.append((torch.tensor(df['label'][i]), df['seq'][i]))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data[index]
    
    
def main(model_state_path, data_path, head_type, freeze):
    os.chdir('/work/home/rnasys/zhouhanwen/nucTran')
    # Tokenizer
    tokenizer_path = 'tokenizer/rnafm/'
    model_max_length = 1026
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=model_max_length, padding_side='left')
    # Config
    hidden_size = 640
    nlabels = 1
    hidden_dropout_prob = 0
    hyperparams = Config(hidden_size, nlabels, hidden_dropout_prob)
    # Inference data
    batch_size = 1
    dataset = MyDataset(data_path)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # Model
    device = torch.device('cuda:0')
    pretrained_state_path = '/work/home/rnasys/zhouhanwen/nucTran/src/RNAFM/RNA-FM-main/src/RNAFM/RNA-FM_pretrained.pth'
    model = RnafmForSequenceClassification(pretrained_weights_location=pretrained_state_path, hyperparams=hyperparams, head_type=head_type, freeze=freeze)    
    model = model.to(device)
    if model_state_path.endswith('.safetensors'):
        load_model(model, filename=model_state_path, strict=True)
    elif model_state_path.endswith('.bin'):
        model.load_state_dict(torch.load(model_state_path), strict=True)
    
    predict_labels, true_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(data):
            labels, batch = batch
            batch = tokenizer(list(batch), return_tensors='pt', padding=True)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            input_ids = input_ids.to(device)
            labels = labels.to(device)
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
            true_labels.extend(labels.tolist())

    result_df = pd.DataFrame({'predict': predict_labels, 'true': true_labels})
    mse = np.mean((np.array(predict_labels) - np.array(true_labels)) ** 2)
    pearson_corr_coef = result_df.corr(method='pearson').iloc[0, 1]
    spearman_corr_coef = result_df.corr(method='spearman').iloc[0, 1]
    
    return mse, pearson_corr_coef, spearman_corr_coef
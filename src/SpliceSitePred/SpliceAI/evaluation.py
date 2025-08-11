from keras.models import load_model
from spliceai.utils import one_hot_encode
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
import json
import tqdm
import os


def predict_ss(seqs, weight_path):
    ys = []
    paths = (weight_path.format(x) for x in range(1, 6))
    models = [load_model(x) for x in paths]
    for seq in tqdm.tqdm(seqs):
        x = one_hot_encode(seq)[None, :] # np.array, [1, length, 4]
        y = np.mean([models[m].predict(x) for m in range(5)], axis=0)[0] # np.array, [1, length, 3]
        ys.append(y)
    ys = np.concatenate(ys)
    return ys


def compute_binary_prauc(true_label, pred_prob):
    """
    Compute PRAUC for single label classification (binary).
    Args:
        true_label(np.array): true labels of sites
        pred_prob(np.array): predicted probabilities of sites, seq len * 1
    """
    precision, recall, _ = precision_recall_curve(true_label, pred_prob)
    prauc = auc(recall, precision)
    return prauc


def compute_ovr_prauc(true_label, pred_prob):
    """
    Compute PRAUC for single label classification (multi-class).
    One vs Rest.
    true_label(np.array): true labels of sites
    pred_prob(np.array): predicted probabilities of sites, seq len * 3
    """
    n_classes = pred_prob.shape[1]
    praucs = []
    for class_idx in range(n_classes):
        prauc = compute_binary_prauc((true_label == class_idx).astype(int), pred_prob[:, class_idx])
        praucs.append(prauc)
    return praucs


def compute_metrics(true_label, pred_prob):
    """
    Compute top-K accuracy for single label (multi-class).
    One vs Rest.
    true_label(np.array): true labels of sites
    pred_prob(np.array): predicted probabilities of sites, seq len * 3
    """
    df = pd.DataFrame(pred_prob)
    df['true_label'] = true_label
    df = df[df["true_label"] != -100]
    counts = df['true_label'].value_counts().to_dict()
    topk_accuracy = [sum((df.sort_values(by=k, ascending=False)[:v])['true_label'] == k) / v for k, v in counts.items()]
    praucs = compute_ovr_prauc(df['true_label'].values, df[[0, 1, 2]].values)
    return topk_accuracy, praucs


def evaluate(weight_path, data_path, output_path, start):
    with open(data_path) as f:
        data = json.load(f)
    
    seqs, labels = [], []
    for i in range(len(data)):
        seqs.append(data[i]['seq'])
        labels.append(data[i]['label'][1:-1])
    
    pred_probs = predict_ss(seqs, weight_path)
    
    true_labels = np.array(labels)[:, start:(start+5000)].flatten()
    topk_accuracy, praucs = compute_metrics(true_labels, pred_probs)
    result_df = pd.DataFrame({'topk_accuracy': topk_accuracy, 'prauc': praucs})
    result_df.to_csv(output_path, index=False)

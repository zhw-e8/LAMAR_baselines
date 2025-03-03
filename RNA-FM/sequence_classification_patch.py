import torch
from torch import nn
import fm
import argparse
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class RnafmForSequenceClassification(nn.Module):

    def __init__(self, pretrained_weights_location, hyperparams, head_type, freeze):
        super().__init__()
        self.hyperparams = hyperparams
        self.freeze = freeze
        
        self.rnafm, self.alphabet = fm.pretrained.rna_fm_t12(model_location=pretrained_weights_location)
        self.classifier = EsmClassificationHead(hyperparams, head_type)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else True

        if self.freeze:
            with torch.no_grad():
                outputs = self.rnafm(
                    input_ids,
                    repr_layers=[12]
                )
        else:
            outputs = self.rnafm(
                input_ids,
                repr_layers=[12]
            )
        sequence_output = outputs["representations"][12] # [batch size, seq len, hidden size]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.hyperparams.num_labels == 1:
                self.problem_type = "regression"
            elif self.hyperparams.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.hyperparams.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.hyperparams.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs["representations"][12],
            attentions=None,
        )
    
    
class Config():
    
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob


class EsmClassificationHead(nn.Module):

    def __init__(self, hyperparams, head_type):
        super().__init__()
        self.head_type = head_type
        if self.head_type == 'Linear':
            self.head = EsmSequenceClassificationLinearHead(hyperparams)
        elif self.head_type == 'CNN':
            self.head = EsmSequenceClassificationCNNHead(hyperparams, kernel_sizes_paddings=[[3, 1], [5, 2], [7, 3]])
        
    def forward(self, features):
        x = self.head(features)
        return x
    

class EsmSequenceClassificationLinearHead(nn.Module):

    def __init__(self, hyperparams):
        super().__init__()
        self.dense = nn.Linear(hyperparams.hidden_size, hyperparams.hidden_size)
        self.dropout = nn.Dropout(hyperparams.hidden_dropout_prob)
        self.out_proj = nn.Linear(hyperparams.hidden_size, hyperparams.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
    
class EsmSequenceClassificationCNNHead(nn.Module):

    def __init__(self, hyperparams, kernel_sizes_paddings):
        """
        kernel_sizes_paddings: such as [[3, 1], [5, 2], [7, 3]]
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=hyperparams.hidden_size, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=False, padding_mode='zeros') for kernel_size, padding in kernel_sizes_paddings]
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hyperparams.hidden_dropout_prob)
        self.out_proj = nn.Linear(128 * len(kernel_sizes_paddings), hyperparams.num_labels)

    def forward(self, features):
        x = features.transpose(1, 2) # features: [batch size, seq len, hidden size], x: [batch size, hidden size, seq len]
        x = [self.relu(conv(x)) for conv in self.convs] # x: [[batch size, out channels, seq len]]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # x: [[batch size, out channels]]
        x = torch.cat(x, dim=1) # x: [batch size, out channels * len(self.convs)]
        x = self.out_proj(x)
        return x

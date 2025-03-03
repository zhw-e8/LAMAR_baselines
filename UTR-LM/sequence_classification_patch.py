from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS
from esm import Alphabet

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from typing import List, Optional, Tuple, Union


class UTRLMForSequenceClassification(nn.Module):

    def __init__(self, hyperparams, head_type, freeze):
        super().__init__()
        self.hyperparams = hyperparams
        self.freeze = freeze
        
        alphabet = Alphabet(mask_prob = 0.0, standard_toks = 'AGCT')
        if hyperparams.model == 'ESM2':
            self.utrlm = ESM2(num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet)
        elif hyperparams.model == 'ESM2_SISS':
            self.utrlm = ESM2_SISS(num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet)
        self.classifier = EsmClassificationHead(hyperparams, head_type)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.hyperparams.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hyperparams.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=self.hyperparams.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

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
                outputs = self.utrlm(
                    input_ids, [6], need_head_weights=True, return_contacts=True, return_representation=True
                )
        else:
            outputs = self.utrlm(
                input_ids, [6], need_head_weights=True, return_contacts=True, return_representation=True
            )
        sequence_output = outputs["representations"][6] # [batch size, seq len, hidden size]
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
            hidden_states=outputs["representations"][6],
            attentions=None,
        )
    
    
class Config():
    
    def __init__(self, model, hidden_size, num_labels, hidden_dropout_prob, initializer_range):
        self.model = model
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range


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
        self.dense = nn.Linear(hyperparams.hidden_size, 40)
        self.dropout = nn.Dropout(hyperparams.hidden_dropout_prob)
        self.dropout3 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(40, hyperparams.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.out_proj(x)
        return x

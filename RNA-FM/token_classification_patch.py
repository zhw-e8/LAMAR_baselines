import torch
from torch import nn
import fm
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class RnafmForTokenClassification(nn.Module):

    def __init__(self, pretrained_weights_location, hyperparams, freeze):
        super().__init__()
        self.hyperparams = hyperparams
        self.freeze = freeze
        
        self.rnafm, self.alphabet = fm.pretrained.rna_fm_t12(model_location=pretrained_weights_location)
        self.classifier = nn.Linear(hyperparams.hidden_size, hyperparams.num_labels)

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


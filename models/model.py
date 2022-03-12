
import torch
import torch.nn as nn 
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    token_logits: torch.FloatTensor = None
    sequence_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForTokenClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        flags=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        token_output = outputs[0]
        sequence_output = outputs[1]

        token_output = self.dropout(token_output)
        token_logits = self.classifier(token_output)

        sequence_logits = self.cls_classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(token_logits.view(-1, self.num_labels), labels.view(-1))
            if flags is not None :
                loss_bn = nn.BCEWithLogitsLoss()
                loss += loss_bn(sequence_logits.squeeze(-1), flags.float())

        if not return_dict:
            output = (token_logits, sequence_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            token_logits=token_logits,
            sequence_logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

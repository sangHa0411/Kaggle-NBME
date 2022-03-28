import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class RdropTrainer(Trainer) :
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        num_labels = model.config.num_labels

        input_names = inputs.keys()
        batch_size = inputs['input_ids'].shape[0]
        labels = inputs.pop('labels')

        for input_name in input_names :
            batch = inputs[input_name]
            inputs[input_name] = torch.cat([batch, batch], dim=0)

        outputs = model(**inputs)

        batch_logits_1 = outputs.logits[:batch_size, :, :]
        batch_logits_2 = outputs.logits[batch_size:, :, :]
        loss_fct_1 = nn.CrossEntropyLoss()
        loss_nll = loss_fct_1(batch_logits_1.view(-1, num_labels), labels.view(-1)) + loss_fct_1(batch_logits_2.view(-1, num_labels), labels.view(-1))

        loss_fct_2 = nn.KLDivLoss(reduction='batchmean')
        loss_kl = loss_fct_2(F.log_softmax(batch_logits_1, dim=-1), F.softmax(batch_logits_2, dim=-1)) + \
            loss_fct_2(F.log_softmax(batch_logits_2, dim=-1), F.softmax(batch_logits_1, dim=-1))
        loss_kl = (loss_kl * 0.1) / 2

        loss = loss_nll + loss_kl
        return (loss, outputs) if return_outputs else loss

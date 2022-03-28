import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Union, Any, Tuple
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

class RdropTrainer(Trainer) :
    def compute_loss(self, model, inputs):
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

        return loss_nll + loss_kl


    def compute_eval_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():               
                if has_labels:
                    with self.autocast_smart_context_manager():
                        loss, outputs = self.compute_eval_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.autocast_smart_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)
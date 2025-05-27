import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer


class CombinedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** 2 * bce).mean()
        return (focal, outputs) if return_outputs else focal


class CustomLossModel(nn.Module):
    def __init__(self, base_model, pos_weight):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)
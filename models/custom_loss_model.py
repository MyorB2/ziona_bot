from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


class CustomLossModel(nn.Module):
    def __init__(self, base_model, pos_weight):
        super().__init__()
        self.base_model = base_model
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(base_model.device))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Remove num_items_in_batch from kwargs before passing to base_model
        kwargs.pop('num_items_in_batch', None)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None
        )

    def gradient_checkpointing_enable(self, **kwargs):
        # Enable gradient checkpointing in the base model
        self.base_model.gradient_checkpointing_enable(**kwargs)

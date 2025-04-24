import torch
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
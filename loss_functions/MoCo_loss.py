import torch.nn.functional

from src.base_classes.abstract_loss import LossFunction


class ContrastiveLoss(LossFunction):
    def __init__(self, pretraining=True):
        self.pretraining = pretraining
        super().__init__()

    def __call__(self, model_out, y, **kwargs):
        if self.pretraining:
            q, logits, zeros = model_out
            loss = torch.nn.functional.cross_entropy(logits, zeros)

        else:
            logits = model_out
            loss = torch.nn.functional.cross_entropy(logits, y)

        return loss
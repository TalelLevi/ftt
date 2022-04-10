from src.base_classes.abstract_metric import Metric
import torch

class accuracy_metric(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, model_out, y, **kwargs):
        logits = model_out
        num_correct = torch.sum(torch.argmax(logits, axis=1) == y).float().item()
        print(num_correct)
        accuracy = (num_correct/len(y)) * 100
        return accuracy


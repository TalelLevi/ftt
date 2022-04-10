
import torch
from train_result import FitResult, BatchResult, EpochResult
from src.base_classes.abstract_trainer import Trainer


# ============== Inherit train Class ==============
class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, metrics=None, device=None):  # TODO add base metric
        super().__init__(model, loss_fn, optimizer, metrics, device)

    def train_batch(self, batch):  # -> BatchResult:
        X, y = batch
        if self.device:
            if self.model.module.pretraining:
                X = [X[0].to(self.device), X[1].to(self.device)]
            else:
                X = X.to(self.device)
            y = y.to(self.device)

        # Train the PyTorch model on one batch of data.
        #  - Forward pass
        #  - Backward pass
        #  - Optimize params
        #  - Calculate accuracy
        self.model.train()

        # forward pass
        # TODO check this
        # if self.model.module.pretraining:
        #     q, logits, zeros = self.model(*X)
        #     loss = self.loss_fn(logits.float(), zeros.long())  # same as calling loss_fn.forward()
        # else:
        #     out = self.model(X)
        #     loss = self.loss_fn(out.float(), y.long())

        model_out = self.model(X)
        loss = self.loss_fn(model_out, y)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # weight update
        self.optimizer.step()

        # calculate the metric
        metrics = [metric(model_out, y) for metric in self.metric]

        # if not self.model.module.pretraining:
        #     # calc accuracy
        #     num_correct = torch.sum(torch.argmax(out, axis=1) == y).float().item()
        #     return BatchResult(loss, num_correct)

        # return BatchResult(loss.item(), metrics)
        return loss.item(), metrics

    def test_batch(self, batch):# -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # Evaluate the PyTorch model on one batch of data.
            #  - Forward pass
            #  - Calculate number of correct predictions
            self.model.train(False)
            y_hat = self.model(X)  # same as calling model.forward()
            loss = self.loss_fn(y_hat, y).item()  # same as calling loss_fn.forward()
            num_correct = torch.sum(torch.argmax(y_hat, axis=1) == y).float().item()

        # calculate the metric
        metrics = [metric(y_hat, y) for metric in self.metric]
        # return BatchResult(loss, num_correct)

        return loss, metrics

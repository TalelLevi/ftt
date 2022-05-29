
import torch
from base_classes.abstract_trainer import Trainer


# ============== Inherit train Class ==============
class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, metrics=None, device=None, logger=None):  # TODO add base metric
        super().__init__(model, loss_fn, optimizer, metrics, device, logger)

    def train_batch(self, batch):  # -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # Train the PyTorch model on one batch of data.
        self.model.train()

        # forward pass
        with torch.cuda.amp.autocast():
            model_out = self.model(X)
            loss = self.loss_fn(model_out, y)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # weight update
        self.optimizer.step()

        # calculate the metric
        metrics = [metric(model_out, y) for metric in self.metric]

        # return BatchResult(loss.item(), metrics)
        return loss.item(), metrics

    def test_batch(self, batch):# -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # Evaluate the PyTorch model on one batch of data.
            self.model.train(False)
            model_out = self.model(X)                   # same as calling model.forward()
            loss = self.loss_fn(model_out, y).item()    # same as calling loss_fn.forward()
            # num_correct = torch.sum(torch.argmax(y_hat, axis=1) == y).float().item()

        # calculate the metric
        metrics = [metric(model_out, y) for metric in self.metric]
        # return BatchResult(loss, num_correct)

        return loss, metrics

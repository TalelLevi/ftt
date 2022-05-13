import abc
import sys

import numpy as np
import tqdm
import torch
from typing import Any, Callable
from torch.utils.data import DataLoader
from pytorch_utils.utils import measure_runtime
from train_result import FitResult, BatchResult, EpochResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, metrics=None, device=None, logger=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param metrics: a list of metrics to apply on the model output
        :param device: torch.device to run training on (CPU or GPU).
        :param logger: clearML logger to track the model
        """
        self.model = model
        self.model_without_ddp = self.model  # .module
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metrics
        self.device = device
        self.logger = logger

        if self.device:
            model.to(self.device)

    def fit(
            self,
            dl_train: DataLoader,
            dl_test: DataLoader,
            num_epochs,
            checkpoint_path: str = None,
            early_stopping: int = None,
            **kw,
    ):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoint_path: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = 0
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            print(f"--- EPOCH {epoch + 1}/{num_epochs} ---")

            epoch_time, res = self.train_epoch(dl_train, **kw)

            self.logger.report_scalar(title=f'Loss per epoch',
                                      series='Loss', value=res.loss, iteration=epoch)

            self.logger.report_scalar(title=f'Accuracy per epoch',
                                      series='Accuracy', value=res.accuracy, iteration=epoch)

            self.logger.report_scalar(title=f'Runtime by epoch',
                                      series='mili-seconds', value=epoch_time, iteration=epoch)

            train_loss.append(res.loss)
            train_acc.append(res.accuracy)

            if not self.model:
                res = self.test_epoch(dl_test, **kw)
                test_loss.append(res.loss)
                test_acc.append(res.accuracy)
            else:
                test_loss.append(res.loss)
                test_acc.append(res.accuracy)
            actual_num_epochs += 1

            # TODO generic the whole early stopping, also add decreasing learning rate
            if res.accuracy > best_acc:
                epochs_without_improvement = 0
                best_acc = res.accuracy
            else:
                epochs_without_improvement += 1

            if checkpoint_path is not None:
                torch.save(self.model, checkpoint_path)

            if early_stopping and epochs_without_improvement >= early_stopping:
                break
            # ========================
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    @measure_runtime
    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        batch_results = np.array([])
        num_batches = len(dl.batch_sampler) if max_batches is None else max_batches

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=sys.stdout) as pbar:

            for batch_idx, data in enumerate(dl):
                if max_batches and batch_idx > max_batches:
                    break

                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res[0]:.3f})")
                pbar.update()

                losses.append(batch_res[0])
                batch_results = np.concatenate((batch_results, np.array(batch_res[1])))


            # calc the metrics for all the batches in the epoch
            avg_loss = np.mean(losses).item()
            accuracy = sum(batch_results) / len(batch_results)
            pbar.set_description(
                f"Epoch metrics:"
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(loss=avg_loss, accuracy=accuracy)

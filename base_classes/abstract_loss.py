import abc


class LossFunction(abc.ABC):
    """
    a basic template class for loss functions
    """
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """
        initiate the loss function class with needed parameters
        """
        pass


    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        implement the loss calculation and return the value
        """
        raise NotImplementedError()


# example
# import torch
# class CrossEntropy(LossFunction):
#     def __init__(self):
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#
#     def __call__(self, y_ture, logits):
#         loss = self.loss_fn(input, target)
#         return loss.item()
#
#
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# loss_fn = CrossEntropy()
# print(loss_fn(input, target))
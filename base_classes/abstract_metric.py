import abc


class Metric(abc.ABC):
    """
    a basic template class for metric evaluation
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
        implement the metric calculation and return the value
        """
        raise NotImplementedError()
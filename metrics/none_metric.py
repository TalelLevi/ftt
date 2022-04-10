from src.base_classes.abstract_metric import Metric


class none_metric(Metric):
    def __init__(self):
        super().__init__()

    def __call__(self, model_out, y, **kwargs):
        return 0


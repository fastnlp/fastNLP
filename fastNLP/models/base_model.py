import torch

from fastNLP.core.trainer import Trainer


class BaseModel(torch.nn.Module):
    """Base PyTorch model for all models.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def fit(self, train_data, dev_data=None, **train_args):
        trainer = Trainer(**train_args)
        trainer.train(self, train_data, dev_data)

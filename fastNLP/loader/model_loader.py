import torch

from fastNLP.loader.base_loader import BaseLoader


class ModelLoader(BaseLoader):
    """
        Loader for models.
    """

    def __init__(self, data_name, data_path):
        super(ModelLoader, self).__init__(data_name, data_path)

    def load_pytorch(self, empty_model):
        """
        Load model parameters from .pkl files into the empty PyTorch model.
        :param empty_model: a PyTorch model with initialized parameters.
        """
        empty_model.load_state_dict(torch.load(self.data_path))

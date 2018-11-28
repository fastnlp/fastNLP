import torch

from fastNLP.io.base_loader import BaseLoader


class ModelLoader(BaseLoader):
    """
        Loader for models.
    """

    def __init__(self):
        super(ModelLoader, self).__init__()

    @staticmethod
    def load_pytorch(empty_model, model_path):
        """
        Load model parameters from .pkl files into the empty PyTorch model.
        :param empty_model: a PyTorch model with initialized parameters.
        :param model_path: str, the path to the saved model.
        """
        empty_model.load_state_dict(torch.load(model_path))

    @staticmethod
    def load_pytorch_model(model_path):
        """Load the entire model.

        """
        return torch.load(model_path)

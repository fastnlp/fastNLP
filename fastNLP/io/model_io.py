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


class ModelSaver(object):
    """Save a model
        Example::
            saver = ModelSaver("./save/model_ckpt_100.pkl")
            saver.save_pytorch(model)

    """

    def __init__(self, save_path):
        """

        :param save_path: str, the path to the saving directory.
        """
        self.save_path = save_path

    def save_pytorch(self, model, param_only=True):
        """Save a pytorch model into .pkl file.

        :param model: a PyTorch model
        :param param_only: bool, whether only to save the model parameters or the entire model.

        """
        if param_only is True:
            torch.save(model.state_dict(), self.save_path)
        else:
            torch.save(model, self.save_path)

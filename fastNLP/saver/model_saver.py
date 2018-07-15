import torch

from fastNLP.saver.base_saver import BaseSaver


class ModelSaver(BaseSaver):
    """Save a models"""

    def __init__(self, save_path):
        super(ModelSaver, self).__init__(save_path)

    def save_pytorch(self, model):
        """
        Save a pytorch model into .pkl file.
        :param model: a PyTorch model
        :return:
        """
        torch.save(model.state_dict(), self.save_path)

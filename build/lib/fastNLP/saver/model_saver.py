import torch


class ModelSaver(object):
    """Save a models"""

    def __init__(self, save_path):
        self.save_path = save_path
        # TODO: check whether the path exist, if not exist, create it.

    def save_pytorch(self, model):
        """
        Save a pytorch model into .pkl file.
        :param model: a PyTorch model
        :return:
        """
        torch.save(model.state_dict(), self.save_path)

import torch


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

    def save_pytorch(self, model):
        """Save a pytorch model into .pkl file.

        :param model: a PyTorch model

        """
        torch.save(model.state_dict(), self.save_path)

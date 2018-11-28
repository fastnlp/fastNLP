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

    def save_pytorch(self, model, param_only=True):
        """Save a pytorch model into .pkl file.

        :param model: a PyTorch model
        :param param_only: bool, whether only to save the model parameters or the entire model.

        """
        if param_only is True:
            torch.save(model.state_dict(), self.save_path)
        else:
            torch.save(model, self.save_path)

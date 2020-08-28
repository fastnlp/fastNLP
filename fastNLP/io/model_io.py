r"""
用于载入和保存模型
"""
__all__ = [
    "ModelLoader",
    "ModelSaver"
]

import torch


class ModelLoader:
    r"""
    用于读取模型
    """
    
    def __init__(self):
        super(ModelLoader, self).__init__()
    
    @staticmethod
    def load_pytorch(empty_model, model_path):
        r"""
        从 ".pkl" 文件读取 PyTorch 模型

        :param empty_model: 初始化参数的 PyTorch 模型
        :param str model_path: 模型保存的路径
        """
        empty_model.load_state_dict(torch.load(model_path))
    
    @staticmethod
    def load_pytorch_model(model_path):
        r"""
        读取整个模型

        :param str model_path: 模型保存的路径
        """
        return torch.load(model_path)


class ModelSaver(object):
    r"""
    用于保存模型
    
    Example::

        saver = ModelSaver("./save/model_ckpt_100.pkl")
        saver.save_pytorch(model)

    """
    
    def __init__(self, save_path):
        r"""

        :param save_path: 模型保存的路径
        """
        self.save_path = save_path
    
    def save_pytorch(self, model, param_only=True):
        r"""
        把 PyTorch 模型存入 ".pkl" 文件

        :param model: PyTorch 模型
        :param bool param_only: 是否只保存模型的参数（否则保存整个模型）

        """
        if param_only is True:
            torch.save(model.state_dict(), self.save_path)
        else:
            torch.save(model, self.save_path)

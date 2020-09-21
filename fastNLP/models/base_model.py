r"""undocumented"""

__all__ = []

import torch

from ..modules.decoder.mlp import MLP


class BaseModel(torch.nn.Module):
    r"""Base PyTorch model for all models.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def fit(self, train_data, dev_data=None, **train_args):
        pass
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class NaiveClassifier(BaseModel):
    r"""
    一个简单的分类器例子，可用于各种测试
    """
    def __init__(self, in_feature_dim, out_feature_dim):
        super(NaiveClassifier, self).__init__()
        self.mlp = MLP([in_feature_dim, in_feature_dim, out_feature_dim])
    
    def forward(self, x):
        return {"predict": torch.sigmoid(self.mlp(x))}
    
    def predict(self, x):
        return {"predict": torch.sigmoid(self.mlp(x)) > 0.5}

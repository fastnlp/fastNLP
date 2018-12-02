import torch


class BaseModel(torch.nn.Module):
    """Base PyTorch model for all models.
    """

    def __init__(self):
        super(BaseModel, self).__init__()

    def fit(self, train_data, dev_data=None, **train_args):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class LinearClassifier(BaseModel):
    def __init__(self, in_feature_dim, out_feature_dim):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(in_feature_dim, out_feature_dim)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        return {"predict": self.softmax(self.linear(x))}

    def predict(self, x):
        return {"predict": self.softmax(self.linear(x))}

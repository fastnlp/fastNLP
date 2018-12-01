import torch


class Optimizer(object):
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not isinstance(model_params, torch.Tensor):
            raise RuntimeError("model parameters should be torch.Tensor, rather than {}".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs


class SGD(Optimizer):
    def __init__(self, model_params=None, lr=0.001, momentum=0.9):
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            self.model_params = model_params
        return torch.optim.SGD(self.model_params, **self.settings)


class Adam(Optimizer):
    def __init__(self, model_params=None, lr=0.001, weight_decay=0.8):
        super(Adam, self).__init__(model_params, lr=lr, weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            self.model_params = model_params
        return torch.optim.Adam(self.model_params, **self.settings)

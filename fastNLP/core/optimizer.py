import torch


class Optimizer(object):
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0, model_params=None):
        """

        :param float lr: learning rate. Default: 0.01
        :param float momentum: momentum. Default: 0
        :param model_params: a generator. E.g. model.parameters() for PyTorch models.
        """
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(model_params, **self.settings)
        else:
            return torch.optim.SGD(self.model_params, **self.settings)


class Adam(Optimizer):
    def __init__(self, lr=0.01, weight_decay=0, model_params=None):
        """

        :param float lr: learning rate
        :param float weight_decay:
        :param model_params: a generator. E.g. model.parameters() for PyTorch models.
        """
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adam, self).__init__(model_params, lr=lr, weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adam(model_params, **self.settings)
        else:
            return torch.optim.Adam(self.model_params, **self.settings)

"""
optimizer 模块定义了 fastNLP 中所需的各种优化器，一般做为 :class:`~fastNLP.Trainer` 的参数使用。

"""
__all__ = [
    "Optimizer",
    "SGD",
    "Adam"
]

import torch


class Optimizer(object):
    """
    别名：:class:`fastNLP.Optimizer` :class:`fastNLP.core.optimizer.Optimizer`

    :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    :param kwargs: additional parameters.
    """
    
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs
    
    def construct_from_pytorch(self, model_params):
        raise NotImplementedError
    
    def _get_require_grads_param(self, params):
        """
        将params中不需要gradient的删除
        :param iterable params: parameters
        :return: list(nn.Parameters)
        """
        return [param for param in params if param.requires_grad]


class SGD(Optimizer):
    """
    别名：:class:`fastNLP.SGD` :class:`fastNLP.core.optimizer.SGD`

    :param float lr: learning rate. Default: 0.01
    :param float momentum: momentum. Default: 0
    :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    """
    
    def __init__(self, lr=0.001, momentum=0, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)
    
    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.SGD(self._get_require_grads_param(self.model_params), **self.settings)


class Adam(Optimizer):
    """
    别名：:class:`fastNLP.Adam` :class:`fastNLP.core.optimizer.Adam`

    :param float lr: learning rate
    :param float weight_decay:
    :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    """
    
    def __init__(self, lr=0.001, weight_decay=0, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adam, self).__init__(model_params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad,
                                   weight_decay=weight_decay)
    
    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adam(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adam(self._get_require_grads_param(self.model_params), **self.settings)

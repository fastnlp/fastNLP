import torch


class Loss(object):
    """Loss function of the algorithm,
    either the wrapper of a loss function from framework, or a user-defined loss (need pytorch auto_grad support)

    """

    def __init__(self, args):
        if args is None:
            # this is useful when
            self._loss = None
        elif isinstance(args, str):
            self._loss = self._borrow_from_pytorch(args)
        else:
            raise NotImplementedError

    def get(self):
        return self._loss

    @staticmethod
    def _borrow_from_pytorch(loss_name):
        if loss_name == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

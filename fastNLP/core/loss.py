import torch


class Loss(object):
    """Loss function of the algorithm,
    either the wrapper of a loss function from framework, or a user-defined loss (need pytorch auto_grad support)

    """

    def __init__(self, args):
        """

        :param args: None or str, the name of a loss function.

        """
        if args is None:
            # this is useful when Trainer.__init__ performs type check
            self._loss = None
        elif isinstance(args, str):
            self._loss = self._borrow_from_pytorch(args)
        else:
            raise NotImplementedError

    def get(self):
        """

        :return self._loss: the loss function
        """
        return self._loss

    @staticmethod
    def _borrow_from_pytorch(loss_name):
        """Given a name of a loss function, return it from PyTorch.

        :param loss_name: str, the name of a loss function
        :return loss: a PyTorch loss
        """
        if loss_name == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

import torch


class Optimizer(object):
    """Wrapper of optimizer from framework

            1. Adam: lr (float), weight_decay (float)
            2. AdaGrad
            3. RMSProp
            4. SGD: lr (float), momentum (float)

    """

    def __init__(self, optimizer_name, **kwargs):
        """
        :param optimizer_name: str, the name of the optimizer
        :param kwargs: the arguments

        """
        self.optim_name = optimizer_name
        self.kwargs = kwargs

    @property
    def name(self):
        """The name of the optimizer.

        :return: str
        """
        return self.optim_name

    @property
    def params(self):
        """The arguments used to create the optimizer.

        :return: dict of (str, *)
        """
        return self.kwargs

    def construct_from_pytorch(self, model_params):
        """Construct a optimizer from framework over given model parameters."""

        if self.optim_name in ["SGD", "sgd"]:
            if "lr" in self.kwargs:
                if "momentum" not in self.kwargs:
                    self.kwargs["momentum"] = 0
                optimizer = torch.optim.SGD(model_params, lr=self.kwargs["lr"], momentum=self.kwargs["momentum"])
            else:
                raise ValueError("requires learning rate for SGD optimizer")

        elif self.optim_name in ["adam", "Adam"]:
            if "lr" in self.kwargs:
                if "weight_decay" not in self.kwargs:
                    self.kwargs["weight_decay"] = 0
                optimizer = torch.optim.Adam(model_params, lr=self.kwargs["lr"],
                                             weight_decay=self.kwargs["weight_decay"])
            else:
                raise ValueError("requires learning rate for Adam optimizer")

        else:
            raise NotImplementedError

        return optimizer

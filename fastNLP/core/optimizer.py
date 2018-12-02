import torch


class Optimizer(object):
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        model_params, lr, momentum = None, 0.01, 0.9
        if len(args) == 0 and len(kwargs) == 0:
            # SGD()
            pass
        elif len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], float) or isinstance(args[0], int):
                # SGD(0.001)
                lr = args[0]
            elif hasattr(args[0], "__next__"):
                # SGD(model.parameters())  args[0] is a generator
                model_params = args[0]
            else:
                raise RuntimeError("Not supported type {}.".format(type(args[0])))
        elif 2 >= len(kwargs) > 0 and len(args) <= 1:
            # SGD(lr=0.01), SGD(lr=0.01, momentum=0.9), SGD(model.parameters(), lr=0.1, momentum=0.9)
            if len(args) == 1:
                if hasattr(args[0], "__next__"):
                    model_params = args[0]
                else:
                    raise RuntimeError("Not supported type {}.".format(type(args[0])))
            if not all(key in ("lr", "momentum") for key in kwargs):
                raise RuntimeError("Invalid SGD arguments. Expect {}, got {}.".format(("lr", "momentum"), kwargs))
            lr = kwargs.get("lr", 0.01)
            momentum = kwargs.get("momentum", 0.9)
        else:
            raise RuntimeError("SGD only accept 0 or 1 sequential argument, but got {}: {}".format(len(args), args))

        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(model_params, **self.settings)
        else:
            return torch.optim.SGD(self.model_params, **self.settings)


class Adam(Optimizer):
    def __init__(self, *args, **kwargs):
        model_params, lr, weight_decay = None, 0.01, 0.9
        if len(args) == 0 and len(kwargs) == 0:
            pass
        elif len(args) == 1 and len(kwargs) == 0:
            if isinstance(args[0], float) or isinstance(args[0], int):
                lr = args[0]
            elif hasattr(args[0], "__next__"):
                model_params = args[0]
            else:
                raise RuntimeError("Not supported type {}.".format(type(args[0])))
        elif 2 >= len(kwargs) > 0 and len(args) <= 1:
            if len(args) == 1:
                if hasattr(args[0], "__next__"):
                    model_params = args[0]
                else:
                    raise RuntimeError("Not supported type {}.".format(type(args[0])))
            if not all(key in ("lr", "weight_decay") for key in kwargs):
                raise RuntimeError("Invalid Adam arguments. Expect {}, got {}.".format(("lr", "weight_decay"), kwargs))
            lr = kwargs.get("lr", 0.01)
            weight_decay = kwargs.get("weight_decay", 0.9)
        else:
            raise RuntimeError("Adam only accept 0 or 1 sequential argument, but got {}: {}".format(len(args), args))

        super(Adam, self).__init__(model_params, lr=lr, weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adam(model_params, **self.settings)
        else:
            return torch.optim.Adam(self.model_params, **self.settings)

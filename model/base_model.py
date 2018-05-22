class BaseModel(object):
    """base model for all models"""

    def __init__(self):
        pass

    def prepare_input(self, data):
        raise NotImplementedError

    def mode(self, test=False):
        raise NotImplementedError

    def data_forward(self, x):
        raise NotImplementedError

    def grad_backward(self):
        raise NotImplementedError

    def loss(self, pred, truth):
        raise NotImplementedError

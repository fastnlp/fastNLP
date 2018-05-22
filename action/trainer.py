from action.action import Action


class Trainer(Action):
    """
        Trainer for common training logic of all models
    """

    def __init__(self, arg):
        super(Trainer, self).__init__()
        self.arg = arg

    def train(self, args):
        raise NotImplementedError

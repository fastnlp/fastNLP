class Action(object):
    """
        base class for Trainer and Tester
    """

    def __init__(self):
        super(Action, self).__init__()

    def load_config(self, args):
        pass

    def load_dataset(self, args):
        pass

    def log(self, args):
        pass

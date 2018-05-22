
class Action(object):
    """
        base class for Trainer and Tester
    """

    def __init__(self):
        super(Action, self).__init__()
        self.logger = None

    def load_config(self, args):
        raise NotImplementedError

    def load_dataset(self, args):
        raise NotImplementedError

    def log(self, args):
        self.logger.log(args)

    """
        Basic operations shared between Trainer and Tester.
    """

    def batchify(self, X, Y=None):
        # a generator
        raise NotImplementedError

    def make_log(self, *args):
        raise NotImplementedError

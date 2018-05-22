
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
        print("call logger.log")

    def batchify(self, X, Y=None):
        """
        :param X:
        :param Y:
        :return iteration:int, the number of step in each epoch
                 generator:generator, to generate batch inputs
        """
        data = X
        if Y is not None:
            data = [X, Y]
        return 2, self._batch_generate(data)

    def _batch_generate(self, data):
        step = 10
        for i in range(2):
            start = i * step
            end = (i + 1) * step
            yield data[0][start:end], data[1][start:end]

    def make_log(self, *args):
        return "log"

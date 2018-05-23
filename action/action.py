from saver.logger import Logger


class Action(object):
    """
        base class for Trainer and Tester
    """

    def __init__(self):
        super(Action, self).__init__()
        self.logger = Logger("logger_output.txt")

    def load_config(self, args):
        raise NotImplementedError

    def load_dataset(self, args):
        raise NotImplementedError

    def log(self, string):
        self.logger.log(string)

    def batchify(self, batch_size, X, Y=None):
        """
        :param batch_size: int
        :param X: feature matrix of size [n_sample, m_feature]
        :param Y: label vector of size [n_sample, 1] (optional)
        :return iteration:int, the number of step in each epoch
                 generator:generator, to generate batch inputs
        """
        n_samples = X.shape[0]
        num_iter = n_samples / batch_size
        if Y is None:
            generator = self._batch_generate(batch_size, num_iter, X)
        else:
            generator = self._batch_generate(batch_size, num_iter, X, Y)
        return num_iter, generator

    @staticmethod
    def _batch_generate(batch_size, num_iter, *data):
        for step in range(num_iter):
            start = batch_size * step
            end = (batch_size + 1) * step
            yield tuple([x[start:end, :] for x in data])

    def make_log(self, *args):
        return "log"

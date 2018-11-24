from collections import defaultdict

import torch

from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler


class Tester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, batch_size, evaluator, use_cuda, save_path="./save/", **kwargs):
        super(Tester, self).__init__()

        self.batch_size = batch_size
        self.pickle_path = save_path
        self.use_cuda = use_cuda
        self._evaluator = evaluator

        self._model = None
        self.eval_history = []  # evaluation results of all batches

    def test(self, network, dev_data):
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
        else:
            self._model = network

        # turn on the testing mode; clean up the history
        self.mode(network, is_test=True)
        self.eval_history.clear()
        output, truths = defaultdict(list), defaultdict(list)
        data_iterator = Batch(dev_data, self.batch_size, sampler=RandomSampler(), as_numpy=False)

        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                prediction = self.data_forward(network, batch_x)
                assert isinstance(prediction, dict)
                for k, v in prediction.items():
                    output[k].append(v)
                for k, v in batch_y.items():
                    truths[k].append(v)
            eval_results = self.evaluate(**output, **truths)
        print("[tester] {}".format(self.print_eval_results(eval_results)))
        self.mode(network, is_test=False)
        self.metrics = eval_results
        return eval_results

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def data_forward(self, network, x):
        """A forward pass of the model. """
        y = network(**x)
        return y

    def evaluate(self, **kwargs):
        """Compute evaluation metrics.
        """
        return self._evaluator(**kwargs)

    def print_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        return ", ".join([str(key) + "=" + str(value) for key, value in results.items()])

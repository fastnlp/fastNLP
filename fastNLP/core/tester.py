import itertools
from collections import defaultdict

import torch
from torch import nn

from fastNLP.core.batch import Batch
from fastNLP.core.sampler import RandomSampler
from fastNLP.core.dataset import DataSet
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import get_func_signature
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.metrics import _prepare_metrics

class Tester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, data, model, metrics, batch_size=16, use_cuda=False, verbose=0):
        super(Tester, self).__init__()

        if not isinstance(data, DataSet):
            raise TypeError(f"The type of data must be `fastNLP.DataSet`, got `{type(data)}`.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.metrics = _prepare_metrics(metrics)

        # check predict
        if hasattr(self._model, 'predict'):
            self._predict_func = self._model.predict
            if not callable(self._predict_func):
                _model_name = model.__class__.__name__
                raise TypeError(f"`{_model_name}.predict` must be callable to be used "
                                f"for evaluation, not `{type(self._predict_func)}`.")
        else:
            self._predict_func = self._model

        self.data = data
        if torch.cuda.is_available() and self.use_cuda:
            self._model = model.cuda()
        else:
            self._model = model
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.verbose = verbose

        self._model_device = model.parameters().__next__().device


    def test(self):
        # turn on the testing mode; clean up the history
        network = self._model
        self.mode(network, is_test=True)
        output, truths = defaultdict(list), defaultdict(list)
        data_iterator = Batch(self.data, self.batch_size, sampler=RandomSampler(), as_numpy=False)

        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                _move_dict_value_to_device(self._model_device, batch_x, batch_y)
                prediction = self.data_forward(network, batch_x)
                assert isinstance(prediction, dict)
                for k, v in prediction.items():
                    output[k].append(v)
                for k, v in batch_y.items():
                    truths[k].append(v)
            for k, v in output.items():
                output[k] = itertools.chain(*v)
            for k, v in truths.items():
                truths[k] = itertools.chain(*v)
            eval_results = {}
            for metric in self.metrics:
                eval_result = metric(output, truths)
                metric_name = metric.__class__.__name__
                eval_results[metric_name] = eval_result
        if self.verbose >= 0:
            print("[tester] \n{}".format(self.format_eval_results(eval_results)))
        self.mode(network, is_test=False)
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
        x = _build_args(network.forward, **x)
        y = self._predict_func(**x)
        return y

    def format_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + '\n\t'
            _str += ", ".join([str(key) + "=" + str(value) for key, value in results.items()])
        _str += '\n'
        return _str

from collections import defaultdict

import torch
from torch import nn

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.utils import CheckError
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_loss_evaluate
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import get_func_signature


class Tester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, data, model, metrics, batch_size=16, use_cuda=False, verbose=1):
        super(Tester, self).__init__()

        if not isinstance(data, DataSet):
            raise TypeError(f"The type of data must be `fastNLP.DataSet`, got `{type(data)}`.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.metrics = _prepare_metrics(metrics)

        self.data = data
        if torch.cuda.is_available() and self.use_cuda:
            self._model = model.cuda()
        else:
            self._model = model
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.verbose = verbose
        self._model_device = model.parameters().__next__().device

        # check predict
        if hasattr(self._model, 'predict'):
            self._predict_func = self._model.predict
            if not callable(self._predict_func):
                _model_name = model.__class__.__name__
                raise TypeError(f"`{_model_name}.predict` must be callable to be used "
                                f"for evaluation, not `{type(self._predict_func)}`.")
        else:
            self._predict_func = self._model.forward

    def test(self):
        # turn on the testing mode; clean up the history
        network = self._model
        self._mode(network, is_test=True)
        output, truths = defaultdict(list), defaultdict(list)
        data_iterator = Batch(self.data, self.batch_size, sampler=SequentialSampler(), as_numpy=False)
        eval_results = {}
        try:
            with torch.no_grad():
                for batch_x, batch_y in data_iterator:
                    _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                    prediction = self._data_forward(self._predict_func, batch_x)
                    if not isinstance(prediction, dict):
                        raise TypeError(f"The return value of {get_func_signature(self._predict_func)} " 
                                                         f"must be `dict`, got {type(prediction)}.")
                    for metric in self.metrics:
                        metric(prediction, batch_y)
                for metric in self.metrics:
                    eval_result = metric.get_metric()
                    if not isinstance(eval_result, dict):
                        raise TypeError(f"The return value of {get_func_signature(metric.get_metric)} must be "
                                        f"`dict`, got {type(eval_result)}")
                    metric_name = metric.__class__.__name__
                    eval_results[metric_name] = eval_result
        except CheckError as e:
            prev_func_signature = get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, output=output, batch_y=truths, check_level=0)

        if self.verbose >= 1:
            print("[tester] \n{}".format(self._format_eval_results(eval_results)))
        self._mode(network, is_test=False)
        return eval_results

    def _mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def _data_forward(self, func, x):
        """A forward pass of the model. """
        x = _build_args(func, **x)
        y = func(**x)
        return y

    def _format_eval_results(self, results):
        """Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]

import torch
from torch import nn

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.utils import _CheckError
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_loss_evaluate
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import _get_func_signature
from fastNLP.core.utils import _get_device


class Tester(object):
    """
    Tester是在提供数据，模型以及metric的情况下进行性能测试的类

    Example::

        import numpy as np
        import torch
        from torch import nn
        from fastNLP import Tester
        from fastNLP import DataSet
        from fastNLP import AccuracyMetric


        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1, 1)
            def forward(self, a):
                return {'pred': self.fc(a.unsqueeze(1)).squeeze(1)}

        model = Model()

        dataset = DataSet({'a': np.arange(10, dtype=float), 'b':np.arange(10, dtype=float)*2})

        dataset.set_input('a')
        dataset.set_target('b')

        tester = Tester(dataset, model, metrics=AccuracyMetric())
        eval_results = tester.test()

    这里Metric的映射规律是和 Trainer_ 中一致的，请参考 Trainer_ 使用metrics。



    """

    def __init__(self, data, model, metrics, batch_size=16, device=None, verbose=1):
        """传入模型，数据以及metric进行验证。

            :param DataSet data: 需要测试的数据集
            :param torch.nn.module model: 使用的模型
            :param MetricBase metrics: 一个Metric或者一个列表的metric对象
            :param int batch_size: evaluation时使用的batch_size有多大。
            :param str,torch.device,None device: 将模型load到哪个设备。默认为None，即Trainer不对模型的计算位置进行管理。支持
                以下的输入str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中,
                可见的第二个GPU中; torch.device，将模型装载到torch.device上。
            :param int verbose: 如果为0不输出任何信息; 如果为1，打印出验证结果。

        """

        super(Tester, self).__init__()

        if not isinstance(data, DataSet):
            raise TypeError(f"The type of data must be `fastNLP.DataSet`, got `{type(data)}`.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")

        self.metrics = _prepare_metrics(metrics)

        self.data = data
        self.device = _get_device(device, check_exist=False)
        self.batch_size = batch_size
        self.verbose = verbose

        if self.device is not None:
            self._model = model.to(self.device)
        else:
            self._model = model
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
        """开始进行验证，并返回验证结果。

        :return dict(dict) eval_results: dict为二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。
            一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
        """
        # turn on the testing mode; clean up the history
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = Batch(self.data, self.batch_size, sampler=SequentialSampler(), as_numpy=False)
        eval_results = {}
        try:
            with torch.no_grad():
                for batch_x, batch_y in data_iterator:
                    _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                    pred_dict = self._data_forward(self._predict_func, batch_x)
                    if not isinstance(pred_dict, dict):
                        raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                        f"must be `dict`, got {type(pred_dict)}.")
                    for metric in self.metrics:
                        metric(pred_dict, batch_y)
                for metric in self.metrics:
                    eval_result = metric.get_metric()
                    if not isinstance(eval_result, dict):
                        raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                        f"`dict`, got {type(eval_result)}")
                    metric_name = metric.__class__.__name__
                    eval_results[metric_name] = eval_result
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)

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

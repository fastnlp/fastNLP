r"""
tester模块实现了 fastNLP 所需的Tester类，能在提供数据、模型以及metric的情况下进行性能测试。

.. code-block::

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

这里Metric的映射规律是和 :class:`fastNLP.Trainer` 中一致的，具体使用请参考 :mod:`trainer 模块<fastNLP.core.trainer>` 的1.3部分。
Tester在验证进行之前会调用model.eval()提示当前进入了evaluation阶段，即会关闭nn.Dropout()等，在验证结束之后会调用model.train()恢复到训练状态。


"""
import time

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except:
    from .utils import _pseudo_tqdm as tqdm

from .batch import BatchIter, DataSetIter
from .dataset import DataSet
from .metrics import _prepare_metrics
from .sampler import SequentialSampler
from .utils import _CheckError
from .utils import _build_args
from .utils import _check_loss_evaluate
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from .utils import _get_model_device
from .utils import _move_model_to_device
from ._parallel_utils import _data_parallel_wrapper
from ._parallel_utils import _model_contains_inner_module
from functools import partial
from ._logger import logger
from .sampler import Sampler

__all__ = [
    "Tester"
]


class Tester(object):
    r"""
    Tester是在提供数据，模型以及metric的情况下进行性能测试的类。需要传入模型，数据以及metric进行验证。
    """
    
    def __init__(self, data, model, metrics, batch_size=16, num_workers=0, device=None, verbose=1, use_tqdm=True,
                 **kwargs):
        r"""
        
        :param ~fastNLP.DataSet,~fastNLP.BatchIter data: 需要测试的数据集
        :param torch.nn.Module model: 使用的模型
        :param ~fastNLP.core.metrics.MetricBase,List[~fastNLP.core.metrics.MetricBase] metrics: 测试时使用的metrics
        :param int batch_size: evaluation时使用的batch_size有多大。
        :param str,int,torch.device,list(int) device: 将模型load到哪个设备。默认为None，即Trainer不对模型
            的计算位置进行管理。支持以下的输入:
    
            1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中,可见的第一个GPU中,可见的第二个GPU中;
    
            2. torch.device：将模型装载到torch.device上。
    
            3. int: 将使用device_id为该值的gpu进行训练
    
            4. list(int)：如果多于1个device，将使用torch.nn.DataParallel包裹model, 并使用传入的device。
    
            5. None. 为None则不对模型进行任何处理，如果传入的model为torch.nn.DataParallel该值必须为None。
    
            如果模型是通过predict()进行预测的话，那么将不能使用多卡(DataParallel)进行验证，只会使用第一张卡上的模型。
        :param int verbose: 如果为0不输出任何信息; 如果为1，打印出验证结果。
        :param bool use_tqdm: 是否使用tqdm来显示测试进度; 如果为False，则不会显示任何内容。
        :param kwargs: 支持传入sampler控制测试顺序
        """
        super(Tester, self).__init__()

        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be `torch.nn.Module`, got `{type(model)}`.")
        
        self.metrics = _prepare_metrics(metrics)
        
        self.data = data
        self._model = _move_model_to_device(model, device=device)
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_tqdm = use_tqdm
        self.logger = logger

        if isinstance(data, DataSet):
            sampler = kwargs.get('sampler', None)
            if sampler is None:
                sampler = SequentialSampler()
            elif not isinstance(sampler, (Sampler, torch.utils.data.Sampler)):
                raise ValueError(f"The type of sampler should be fastNLP.BaseSampler or pytorch's Sampler, got {type(sampler)}")
            if hasattr(sampler, 'set_batch_size'):
                sampler.set_batch_size(batch_size)
            self.data_iterator = DataSetIter(dataset=data, batch_size=batch_size, sampler=sampler,
                                             num_workers=num_workers)
        elif isinstance(data, BatchIter):
            self.data_iterator = data
        else:
            raise TypeError("data type {} not support".format(type(data)))

        # check predict
        if (hasattr(self._model, 'predict') and callable(self._model.predict)) or \
                (_model_contains_inner_module(self._model) and hasattr(self._model.module, 'predict') and
                 callable(self._model.module.predict)):
            if isinstance(self._model, nn.DataParallel):
                self._predict_func_wrapper = partial(_data_parallel_wrapper('predict',
                                                                    self._model.device_ids,
                                                                    self._model.output_device),
                                                     network=self._model.module)
                self._predict_func = self._model.module.predict  # 用于匹配参数
            elif isinstance(self._model, nn.parallel.DistributedDataParallel):
                self._predict_func = self._model.module.predict
                self._predict_func_wrapper = self._model.module.predict  # 用于调用
            else:
                self._predict_func = self._model.predict
                self._predict_func_wrapper = self._model.predict
        else:
            if _model_contains_inner_module(model):
                self._predict_func_wrapper = self._model.forward
                self._predict_func = self._model.module.forward
            else:
                self._predict_func = self._model.forward
                self._predict_func_wrapper = self._model.forward
    
    def test(self):
        r"""开始进行验证，并返回验证结果。

        :return Dict[Dict]: dict的二层嵌套结构，dict的第一层是metric的名称; 第二层是这个metric的指标。一个AccuracyMetric的例子为{'AccuracyMetric': {'acc': 1.0}}。
        """
        # turn on the testing mode; clean up the history
        self._model_device = _get_model_device(self._model)
        network = self._model
        self._mode(network, is_test=True)
        data_iterator = self.data_iterator
        eval_results = {}
        try:
            with torch.no_grad():
                if not self.use_tqdm:
                    from .utils import _pseudo_tqdm as inner_tqdm
                else:
                    inner_tqdm = tqdm
                with inner_tqdm(total=len(data_iterator), leave=False, dynamic_ncols=True) as pbar:
                    pbar.set_description_str(desc="Test")

                    start_time = time.time()

                    for batch_x, batch_y in data_iterator:
                        _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                        pred_dict = self._data_forward(self._predict_func, batch_x)
                        if not isinstance(pred_dict, dict):
                            raise TypeError(f"The return value of {_get_func_signature(self._predict_func)} "
                                            f"must be `dict`, got {type(pred_dict)}.")
                        for metric in self.metrics:
                            metric(pred_dict, batch_y)

                        if self.use_tqdm:
                            pbar.update()

                    for metric in self.metrics:
                        eval_result = metric.get_metric()
                        if not isinstance(eval_result, dict):
                            raise TypeError(f"The return value of {_get_func_signature(metric.get_metric)} must be "
                                            f"`dict`, got {type(eval_result)}")
                        metric_name = metric.get_metric_name()
                        eval_results[metric_name] = eval_result
                    pbar.close()
                    end_time = time.time()
                    test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
                    if self.verbose >= 0:
                        self.logger.info(test_str)
        except _CheckError as e:
            prev_func_signature = _get_func_signature(self._predict_func)
            _check_loss_evaluate(prev_func_signature=prev_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=self.data, check_level=0)
        finally:
            self._mode(network, is_test=False)
        if self.verbose >= 1:
            logger.info("[tester] \n{}".format(self._format_eval_results(eval_results)))
        return eval_results
    
    def _mode(self, model, is_test=False):
        r"""Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()
    
    def _data_forward(self, func, x):
        r"""A forward pass of the model. """
        x = _build_args(func, **x)
        y = self._predict_func_wrapper(**x)
        return y
    
    def _format_eval_results(self, results):
        r"""Override this method to support more print formats.

        :param results: dict, (str: float) is (metrics name: value)

        """
        _str = ''
        for metric_name, metric_result in results.items():
            _str += metric_name + ': '
            _str += ", ".join([str(key) + "=" + str(value) for key, value in metric_result.items()])
            _str += '\n'
        return _str[:-1]

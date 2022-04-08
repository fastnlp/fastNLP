__all__ = [
    'func_post_proc'
]

from typing import Any
from functools import wraps
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.utils import _module_available

_IS_TORCHMETRICS_AVAILABLE = _module_available('torchmetrics')
if _IS_TORCHMETRICS_AVAILABLE:
    from torchmetrics import Metric as torchmetrics_Metric

_IS_ALLENNLP_AVAILABLE = _module_available('allennlp')
if _IS_ALLENNLP_AVAILABLE:
    from allennlp.training.metrics import Metric as allennlp_Metric

if _NEED_IMPORT_PADDLE:
    from paddle.metric import Metric as paddle_Metric


def _is_torchmetrics_metric(metric: Any) -> bool:
    """
    检查输入的对象是否为torchmetrics对象

    :param metric:
    :return:
    """
    if _IS_TORCHMETRICS_AVAILABLE:
        return isinstance(metric, torchmetrics_Metric)
    else:
        return False


def _is_allennlp_metric(metric: Any) -> bool:
    """
    检查输入的对象是否为allennlp对象

    :param metric:
    :return:
    """
    if _IS_ALLENNLP_AVAILABLE:
        return isinstance(metric, allennlp_Metric)
    else:
        return False


def _is_paddle_metric(metric: Any) -> bool:
    """
    检查输入的对象是否为allennlp对象

    :param metric:
    :return:
    """
    if _NEED_IMPORT_PADDLE:
        return isinstance(metric, paddle_Metric)
    else:
        return False


def func_post_proc(metric: 'Metric', fn: callable, method_name: str) -> 'Metric':
    """
    将fn函数作用包裹在 metric 对象的 {method_name} 方法上，使得 metric.{method_name} 函数的返回结果先经过 fn 函数处理
        后再返回。注意对 metric 的 {method_name} 函数的修改是 inplace 的。

    :param metric: metric对象
    :param fn: 作用于 metric 的 accumulate 方法的返回值
    :param method_name: 一般来说，对于
    :return: metric
    """
    assert hasattr(metric, method_name) and callable(getattr(metric, method_name)), \
        f"Parameter `metric` must have a {method_name} function."
    assert callable(fn), "Parameter `fn` must be callable."

    func = getattr(metric, method_name)

    @wraps(func)
    def wrap_method(*args, **kwargs):
        res = func(*args, **kwargs)
        return fn(res)

    wrap_method.__wrapped_by_func_post_proc__ = True
    setattr(metric, method_name, wrap_method)
    return metric


class AggregateMethodError(BaseException):
    def __init__(self, should_have_aggregate_method, only_warn=False):
        super(AggregateMethodError, self).__init__(self)
        self.should_have_aggregate_method = should_have_aggregate_method
        self.only_warn = only_warn

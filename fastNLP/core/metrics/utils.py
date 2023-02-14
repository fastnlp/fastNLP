from typing import Any

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.utils import _module_available

__all__ = []  # type: ignore

_IS_TORCHMETRICS_AVAILABLE = _module_available('torchmetrics')
_IS_ALLENNLP_AVAILABLE = _module_available('allennlp')
_IS_TORCHEVAL_AVAILABLE = _module_available('torcheval')
if _IS_ALLENNLP_AVAILABLE:
    try:
        from allennlp.training.metrics import Metric as allennlp_Metric
    except Exception:
        # 由于allennlp经常 import 不成功，这里通过这个来规避下风险
        _IS_ALLENNLP_AVAILABLE = False

if _IS_TORCHMETRICS_AVAILABLE:
    from torchmetrics import Metric as torchmetrics_Metric

if _NEED_IMPORT_PADDLE:
    from paddle.metric import Metric as paddle_Metric

if _IS_TORCHEVAL_AVAILABLE:
    from torcheval.metrics import Metric as torcheval_Metric


def _is_torchmetrics_metric(metric: Any) -> bool:
    """检查输入的对象是否为torchmetrics对象。

    :param metric:
    :return:
    """
    if _IS_TORCHMETRICS_AVAILABLE:
        return isinstance(metric, torchmetrics_Metric)
    else:
        return False


def _is_allennlp_metric(metric: Any) -> bool:
    """检查输入的对象是否为allennlp对象。

    :param metric:
    :return:
    """
    if _IS_ALLENNLP_AVAILABLE:
        return isinstance(metric, allennlp_Metric)
    else:
        return False


def _is_paddle_metric(metric: Any) -> bool:
    """检查输入的对象是否为allennlp对象。

    :param metric:
    :return:
    """
    if _NEED_IMPORT_PADDLE:
        return isinstance(metric, paddle_Metric)
    else:
        return False


def _is_torcheval_metric(metric: Any) -> bool:
    """检查输入的对象是否为torcheval对象。

    :param metric:
    :return:
    """
    if _IS_TORCHEVAL_AVAILABLE:
        return isinstance(metric, torcheval_Metric)
    else:
        return False


class AggregateMethodError(BaseException):

    def __init__(self, should_have_aggregate_method, only_warn=False):
        super(AggregateMethodError, self).__init__(self)
        self.should_have_aggregate_method = should_have_aggregate_method
        self.only_warn = only_warn


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec

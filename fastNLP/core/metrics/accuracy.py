__all__ = [
    'Accuracy',
    "TransformersAccuracy"
]

from typing import Union

import numpy as np

from fastNLP.core.metrics.metric import Metric
from fastNLP.core.metrics.backend import Backend
from fastNLP.core.utils.seq_len_to_mask import seq_len_to_mask
from fastNLP.core.log import logger


class Accuracy(Metric):
    """
    计算 准确率 的 metric 。

    :param backend: 目前支持五种类型的backend, ``['auto', 'torch', 'paddle', 'jittor', 'oneflow']``。其中 ``'auto'`` 表示根据实际调用
        :meth:`update` 函数时传入的参数决定具体的 backend ，一般情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字聚合后再得到 metric，
        当 ``backend`` 不支持分布式时，该参数无意义。如果为 ``None`` ，将在 :class:`~fastNLP.core.controllers.Evaluator` 
        中根据 ``sampler`` 是否使用分布式进行自动设置。
    """
    def __init__(self, backend: Union[str, Backend, None] = 'auto', aggregate_when_get_metric: bool = None):
        super(Accuracy, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)
        self.register_element(name='correct', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='total', value=0, aggregate_method="sum", backend=backend)

    def get_metric(self) -> dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计算最终的评价结果。

        :return: 包含以下内容的字典：``{"acc": float, 'total': float, 'correct': float}``；
        """
        evaluate_result = {'acc': round(self.correct.get_scalar() / (self.total.get_scalar() + 1e-12), 6),
                           'total': self.total.item(), 'correct': self.correct.item()}
        return evaluate_result

    def update(self, pred, target, seq_len=None):
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。

        :param pred: 预测的 tensor, tensor 的形状可以是 ``[B,]`` 、``[B, n_classes]`` 、
                ``[B, max_len]`` 或 ``[B, max_len, n_classes]``
        :param target: 真实值的 tensor, tensor 的形状可以是 ``[B,]`` 、``[B,]`` 、``[B, max_len]``
            或 ``[B, max_len]``
        :param seq_len: 序列长度标记, 标记的形状可以是 ``None``,  或者 ``[B]`` 。
                如果 mask 也被传进来的话 ``seq_len`` 会被忽略
        """
        # 为了兼容不同框架，我们将输入变量全部转为numpy类型来进行计算。
        pred = self.tensor2numpy(pred)
        target = self.tensor2numpy(target)
        if seq_len is not None:
            seq_len = self.tensor2numpy(seq_len)

        if seq_len is not None and target.ndim > 1:
            max_len = target.shape[1]
            masks = seq_len_to_mask(seq_len, max_len)
        else:
            masks = None

        if pred.ndim == target.ndim:
            if np.prod(pred.shape) != np.prod(target.shape):
                raise RuntimeError(f"when pred have same dimensions with target, they should have same element numbers."
                                   f" while target have shape:{target.shape}, "
                                   f"pred have shape: {pred.shape}")

        elif pred.ndim == target.ndim + 1:
            pred = pred.argmax(axis=-1)
            if seq_len is None and target.ndim > 1:
                logger.warning("You are not passing `seq_len` to exclude pad when calculate accuracy.")

        else:
            raise RuntimeError(f"when pred have size:{pred.shape}, target should have size: {pred.shape} or "
                               f"{pred.shape[:-1]}, got {target.shape}.")

        if masks is not None:
            self.total += masks.sum().item()
            self.correct += ((pred == target) * masks).sum().item()
        else:
            self.total += np.prod(list(pred.shape)).item()
            self.correct += (target == pred).sum().item()


class TransformersAccuracy(Accuracy):
    """
    适配 :mod:`transformers` 中相关模型的 Accuracy metric 。
    """
    def update(self, logits, labels, attention_mask=None):
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。

        :param logits: 形状为 ``[B, n_classes]`` 或 ``[B, max_len, n_classes]`` 。
        :param labels: 形状为 ``[B, ]`` 或 ``[B, max_len]``
        :param attention_mask: 序列长度标记。
        """
        seq_len = attention_mask.sum(dim=-1)
        super().update(pred=logits, target=labels, seq_len=seq_len)
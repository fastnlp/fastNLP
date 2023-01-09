__all__ = ['Perplexity']

from typing import Any, Optional, Union

import numpy as np
from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.metric import Metric


def softmax(x):
    # 计算每个元素的指数
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    # 返回指数的和
    return exps / np.sum(exps, axis=1, keepdims=True)


class Perplexity(Metric):
    """计算 perplexity 的 metric 。

    :param ignore_label: 指定要忽略的目标类的整数。如果给定，则该类索引不起作用。例如单词表中代表未知单词的目标整数。
    :param backend: 目前支持五种类型的backend,
        ``['auto', 'torch', 'paddle', 'jittor', 'oneflow']``。
        其中 ``'auto'`` 表示根据实际调用。
        :meth:`update` 函数时传入的参数决定具体的 backend ，一般情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字
        聚合后再得到 metric，当 ``backend`` 不支持分布式时，该参数无意义。
        如果为 ``None`` ，将在 :class:`~fastNLP.core.controllers.Evaluator`
        中根据 ``sampler`` 是否使用分布式进行自动设置。
    """

    def __init__(self,
                 ignore_label: Optional[int] = None,
                 backend: Union[str, Backend, None] = 'auto',
                 aggregate_when_get_metric: bool = None,
                 **kwargs: Any):
        super().__init__(backend=backend,
                         aggregate_when_get_metric=aggregate_when_get_metric)
        if ignore_label is not None and not isinstance(ignore_label, int):
            raise ValueError(
                f'Argument `ignore_label` expected to either be `None` or an `int` but got {ignore_label}'
            )
        self.ignore_label = ignore_label
        self.register_element(name='total',
                              value=0.,
                              aggregate_method='sum',
                              backend=backend)
        self.register_element(name='count',
                              value=0.,
                              aggregate_method='sum',
                              backend=backend)

    def update(self, pred, target) -> None:
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。
        :param pred: 分配给序列中每个单词的概率，shape为[batch_size, seq_len, vocab_size]。
        :param target: 序列的真实标签值，shape为[batch_size, seq_len]。
        """
        if len(pred.shape) != 3:
            raise ValueError(
                'Input tensor `pred` is expected to have 3 dimensions, [batch_size, seq_len, vocab_size],'
                f' but got {len(pred.shape)}.')
        if len(target.shape) != 2:
            raise ValueError(
                'Input tensor `target` is expected to have 2 dimensions, [batch_size, seq_len],'
                f' but got {len(target.shape)}.')
        if pred.shape[:2] != target.shape:
            raise ValueError(
                'Input tensors `pred` and `target` are expected to have equaling first two dimensions,'
                f' [batch_size, seq_len], but got {pred.shape[:2]} and {target.shape}.'
            )
        pred = self.tensor2numpy(pred)
        target = self.tensor2numpy(target)
        probs = softmax(pred.reshape(-1, pred.shape[-1]))
        target = target.reshape(-1)
        if self.ignore_label is not None:
            mask = np.not_equal(target, self.ignore_label)
            target = np.ma.array(target, mask=(mask == False),
                                 fill_value=0).filled()
            probs = np.take(probs, target, axis=1).diagonal()[mask]
        else:
            probs = np.take(probs, target, axis=1).diagonal()
        self.total += -np.sum(np.log(probs))
        self.count += probs.size

    def get_metric(self) -> dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计算最终的评价结果。

        :return: 包含以下内容的字典：``{"perplexity": float}``；
        """
        perplexity = np.exp(self.total / self.count)
        result = {'perplexity': round(perplexity, 6)}
        return result

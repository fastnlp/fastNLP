__all__ = [
    'ClassifyFPreRecMetric'
]

from typing import Union, List
from collections import Counter
import numpy as np

from .metric import Metric
from .backend import Backend
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.utils.seq_len_to_mask import seq_len_to_mask
from .utils import _compute_f_pre_rec
from fastNLP.core.log import logger

class ClassifyFPreRecMetric(Metric):
    def __init__(self, tag_vocab: Vocabulary = None, ignore_labels: List[str] = None,
                 only_gross: bool = True, f_type='micro', beta=1, backend: Union[str, Backend, None] = 'auto',
                 aggregate_when_get_metric: bool = None) -> None:
        """

        :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` . 默认值为 ``None``。若为 ``None`` 则使用数字来作为标签内容，
        否则使用 vocab 来作为标签内容。
        :param ignore_labels: ``str`` 组成的 ``list``. 这个 ``list``中的 class 不会被用于计算。例如在 POS tagging 时传入 ``['NN']``，
        则不会计算 'NN' 个 label
        :param only_gross: 是否只计算总的 ``f1``, ``precision``, ``recall``的值；如果为 ``False``，不仅返回总的 ``f1``, ``pre``,
        ``rec``, 还会返回每个 label 的 ``f1``, ``pre``, ``rec``
        :param f_type: `micro` 或 `macro` .
            * `micro` : 通过先计算总体的 TP，FN 和 FP 的数量，再计算 f, precision, recall;
            * `macro` : 分布计算每个类别的 f, precision, recall，然后做平均（各类别 f 的权重相同）
        :param beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` .
        :param backend: 目前支持四种类型的 backend, ``[torch, paddle, jittor, 'auto']``。其中 ``'auto'`` 表示根据实际调用 Metric.update()
        函数时传入的参数决定具体的 backend ，大部分情况下直接使用 ``'auto'`` 即可。
        :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字聚合后再得到metric，
            当 backend 不支持分布式时，该参数无意义。如果为 ``None`` ，将在 Evaluator 中根据 sampler 是否使用分布式进行自动设置。

        """
        super(ClassifyFPreRecMetric, self).__init__(backend=backend,
                                                    aggregate_when_get_metric=aggregate_when_get_metric)
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))
        if tag_vocab:
            if not isinstance(tag_vocab, Vocabulary):
                raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross

        self.tag_vocab = tag_vocab

        self._tp = Counter()
        self._fp = Counter()
        self._fn = Counter()

    def reset(self):
        """
        重置 tp, fp, fn 的值

        """
        # 由于不是 element 了，需要自己手动清零一下
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()

    def get_metric(self) -> dict:
        r"""
        get_metric 函数将根据 update 函数累计的评价指标统计量来计算最终的评价结果.

        :return evaluate_result: {"acc": float}
        """
        evaluate_result = {}

        # 通过 all_gather_object 将各个卡上的结果收集过来，并加和。
        ls = self.all_gather_object([self._tp, self._fp, self._fn])
        tps, fps, fns = zip(*ls)
        _tp, _fp, _fn = Counter(), Counter(), Counter()
        for c, cs in zip([_tp, _fp, _fn], [tps, fps, fns]):
            for _c in cs:
                c.update(_c)

        if not self.only_gross or self.f_type == 'macro':
            tags = set(_fn.keys())
            tags.update(set(_fp.keys()))
            tags.update(set(_tp.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                if self.tag_vocab is not None:
                    tag_name = self.tag_vocab.to_word(tag)
                else:
                    tag_name = int(tag)
                tp = _tp[tag]
                fn = _fn[tag]
                fp = _fp[tag]
                if tp == fn == fp == 0:
                    continue
                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag_name)
                    pre_key = 'pre-{}'.format(tag_name)
                    rec_key = 'rec-{}'.format(tag_name)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square, sum(_tp.values()), sum(_fn.values()), sum(_fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def update(self, pred, target, seq_len=None):
        r"""
        update 函数将针对一个批次的预测结果做评价指标的累计

        :param pred: 预测的 tensor, tensor 的形状可以是 [B,], [B, n_classes])
                [B, max_len], 或者 [B, max_len, n_classes]
        :param target: 真实值的 tensor, tensor 的形状可以是 [B,],
                [B,], [B, max_len], 或者 [B, max_len]
        :param seq_len: 序列长度标记, 标记的形状可以是 None, [B].

        """
        pred = self.tensor2numpy(pred)
        target = self.tensor2numpy(target)
        if seq_len is not None:
            seq_len = self.tensor2numpy(seq_len)

        if seq_len is not None and target.ndim > 1:
            max_len = target.shape[-1]
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = np.ones_like(target)

        if pred.ndim == target.ndim:
            if len(pred.flatten()) != len(target.flatten()):
                raise RuntimeError(f"when pred have same dimensions with target, they should have same element numbers."
                                   f" while target have element numbers:{len(pred.flatten())}, "
                                   f"pred have element numbers: {len(target.flatten())}")

        elif pred.ndim == target.ndim + 1:
            pred = pred.argmax(axis=-1)
            if seq_len is None and target.ndim > 1:
                logger.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"when pred have "
                               f"size:{pred.shape}, target should have size: {pred.shape} or "
                               f"{pred.shape[:-1]}, got {target.shape}.")

        target_idxes = set(target.reshape(-1).tolist()+pred.reshape(-1).tolist())
        for target_idx in target_idxes:
            self._tp[target_idx] += ((pred == target_idx) * (target == target_idx) * masks).sum().item()
            self._fp[target_idx] += ((pred == target_idx) * (target != target_idx) * masks).sum().item()
            self._fn[target_idx] += ((pred != target_idx) * (target == target_idx) * masks).sum().item()

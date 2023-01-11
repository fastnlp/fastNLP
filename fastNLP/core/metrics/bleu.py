__all__ = ['BLEU']

from collections import Counter
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.metric import Metric
from fastNLP.core.utils.utils import _check_valid_parameters_number


def get_n_gram(token: Sequence[str], n_gram: int) -> Counter:
    counter: Counter = Counter()
    for i in range(1, n_gram + 1):
        for j in range(len(token) - i + 1):
            key = tuple(token[j:(i + j)])
            counter[key] += 1
    return counter


def get_tokenizer(lang):
    if lang == 'en':
        return str.split
    elif lang in ('cn', 'zh'):
        return list
    else:
        return None


def _get_brevity_penalty(pred_len: float,
                         references_len: float) -> float:
    if pred_len >= references_len:
        return float(1.)
    elif pred_len == 0 or references_len == 0:
        return float(0.)
    return np.exp(1 - references_len / pred_len)


class BLEU(Metric):
    """计算 bleu 的 metric 。

    :param n_gram: Gram的范围是[1,4]
    :param smooth: 是否选择smoothing计算
    :param ngram_weights: 用来控制各个i-gram所计算结果的权重，sum(nrgam_weights)的结果为 1
    :param backend: 目前支持五种类型的backend, ``['auto', 'torch', 'paddle', 'jittor', 'oneflow']``。
        其中 ``'auto'`` 表示根据实际调用。
        :meth:`update` 函数时传入的参数决定具体的 backend ，一般情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字
        聚合后再得到 metric，当 ``backend`` 不支持分布式时，该参数无意义。
        如果为 ``None`` ，将在 :class:`~fastNLP.core.controllers.Evaluator`
        中根据 ``sampler`` 是否使用分布式进行自动设置。
    :param tokenizer: 用户可以传入Callable函数进行分词
        如果是``str``，则按照传入的语言进行分词，默认选项有['en','cn','zh'],``en``代表英语，其他代表中文
        如果是None，则会再第一次update时选择第一个sample的语言进行选择
    """

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        ngram_weights: Optional[Sequence[float]] = None,
        backend: Union[str, Backend, None] = 'auto',
        aggregate_when_get_metric: bool = None,
        tokenizer_fn: Union[Callable, str] = None,
        **kwargs: Any,
    ):
        super().__init__(backend=backend,
                         aggregate_when_get_metric=aggregate_when_get_metric)
        self.n_gram = n_gram
        self.smooth = smooth
        if ngram_weights is not None and len(ngram_weights) != n_gram:
            raise ValueError(
                f'The number of weights in weights is different from n_gram: {len(ngram_weights)} != {n_gram}'
            )
        self.ngram_weights = ngram_weights if ngram_weights is not None else [
            1.0 / n_gram
        ] * n_gram

        self.register_element(name='pred_len',
                              value=0,
                              aggregate_method='sum',
                              backend=backend)
        self.register_element(name='references_len',
                              value=0,
                              aggregate_method='sum',
                              backend=backend)
        self.register_element(name='precision_matches',
                              value=[0 for _ in range(self.n_gram)],
                              aggregate_method='sum',
                              backend=backend)
        self.register_element(name='precision_total',
                              value=[0 for _ in range(self.n_gram)],
                              aggregate_method='sum',
                              backend=backend)
        if callable(tokenizer_fn):
            _check_valid_parameters_number(tokenizer_fn,
                                           ['text'])  # 检查是否一定是吃进去一个参数
            self.tokenizer_fn = tokenizer_fn
        elif isinstance(tokenizer_fn, str):
            self.tokenizer_fn = get_tokenizer(tokenizer_fn)
            if self.tokenizer_fn is None:
                raise ValueError(
                    "Right now, `tokenizer_fn` only supports pre-defined 'en' or 'cn'."
                )
        else:
            assert tokenizer_fn is None, f'`tokenizer_fn` supports Callable, str or None, but not `{type(tokenizer_fn)}`'
            self.tokenizer_fn = tokenizer_fn

    def update(
        self, predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[str],
                          Sequence[Sequence[str]]]) -> None:
        r"""
       :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。
       :param predictions: 预测的 ``sentence``, type为``Sequence``，长度可变，假设为 ``L``
           * predictions可以为str类型，也可以为list类型。
       :param references: 答案译文，type为``Sequence``，长度必须也为``L``，
           保持和``predictions``一致，每一个元素也是一个``Sequence``。
           * references可以为str类型，但是该情况下predictions也必须为str类型。
           * references可以为list[str]类型，如果predictions只有一条数据，references数量不受限制，
                如果predictions数量超过一条，references的长度必须匹配predictions的数量。
       """

        if isinstance(references, list) and all(
                isinstance(reference, str) for reference in references):
            if isinstance(predictions, str):
                references = [references]
            else:
                if len(predictions) == 1:
                    references = [references]
                else:
                    references = [[reference] for reference in references]

        if isinstance(predictions, str):
            predictions = [predictions]

        if isinstance(references, str):
            references = [[references]]
        assert len(predictions) == len(
            references
        ), 'The number of predictions and references must be equal'

        if self.tokenizer_fn is None:
            lang = 'en'
            for _char in predictions[0]:
                if '\u4e00' <= _char <= '\u9fa5':
                    lang = 'cn'
                    break
            self.tokenizer_fn = get_tokenizer(lang)
        references_token: Sequence[Sequence[Sequence[str]]] = [
            [self.tokenizer_fn(line) for line in r] for r in references
        ]
        predictions_token: Sequence[Sequence[str]] = [
            self.tokenizer_fn(line) for line in predictions
        ]
        for prediction, references in zip(predictions_token, references_token):
            self.pred_len += len(prediction)
            self.references_len += len(
                min(references, key=lambda x: abs(len(x) - len(prediction))))
            pred_counter: Counter = get_n_gram(prediction, self.n_gram)
            reference_counter: Counter = Counter()
            for reference in references:
                reference_counter |= get_n_gram(reference, self.n_gram)

            counter_clip = pred_counter & reference_counter

            for ngram in counter_clip:
                self.precision_matches[len(ngram) - 1] += counter_clip[ngram]
            for ngram in pred_counter:
                self.precision_total[len(ngram) - 1] += pred_counter[ngram]

    def get_metric(self) -> dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计算最终的评价结果。

        :return: 包含以下内容的字典：``{"bleu": float}``；
        """

        precision_matches = self.precision_matches.tensor2numpy()
        precision_total = self.precision_total.tensor2numpy()
        if min(precision_matches) == 0.0:
            return {'bleu': float(0.0)}
        if self.smooth:
            precision_score = (precision_matches + 1) / (precision_total + 1)
            precision_score[0] = precision_matches[0] / precision_total[0]
        else:
            precision_score = precision_matches / precision_total

        precision_score = np.array(
            self.ngram_weights) * np.log(precision_score)
        brevity_penalty = _get_brevity_penalty(
            self.pred_len.get_scalar(), self.references_len.get_scalar())
        bleu = brevity_penalty * np.exp(np.sum(precision_score))
        result = {'bleu': round(float(bleu), 6)}
        return result

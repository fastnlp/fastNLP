import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.metric import Metric
from fastNLP.core.utils.utils import _check_valid_parameters_number
from fastNLP.envs.utils import _module_available

__all__ = ['ROUGE']


def get_tokenizer(lang):
    if lang == 'en':
        return str.split
    elif lang in ('cn', 'zh'):
        return list
    else:
        return None


def _normalize_and_tokenize(
    text: str,
    tokenizer: Callable[[str], Sequence[str]],
    stemmer: Optional[Any] = None,
    normalizer: Optional[Callable[[str], str]] = None,
) -> Sequence[str]:
    r"""对``sentence``使用Porter stemmer用于去除单词后缀以改进匹配。并规范化句子以
    及进行分词。

    :param text: 一个输入的句子.
    :param stemmer: Porter-stemmer实例来去除单词后缀以改进匹配。
    :param normalizer: 用户自己的规范化函数。
        如果这值是``None``，则默认使用空格替换任何非字母数字字符。
        这个函数必须输入一个 ``str`` 并且返回 ``str``。
    :param tokenizer: 分词函数，用户传入的函数，或者是类中包含的中英文分词函数。
    """
    if tokenizer == str.split:
        text = normalizer(text) if callable(normalizer) else re.sub(
            r'[^a-z0-9]+', ' ', text.lower())
    tokens = tokenizer(text)
    if stemmer:
        tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]
    tokens = [x for x in tokens if (isinstance(x, str) and len(x) > 0)]
    return tokens


def _compute_metrics(matches: int, pred_len: int,
                     reference_len: int) -> Dict[str, float]:
    r"""这个函数将根据命中数或者LCS和列表的长度去计算预测句子和标准译文句子的精度、召
    回率和F1得分值。

    :param matches: 匹配数或最长公共子序列的长度。
    :param pred_len: 预测句子的序列长度。
    :param reference_len: 答案句子的序列长度。
    """
    precision = matches / pred_len
    recall = matches / reference_len
    if precision == recall == 0.0:
        return dict(
            precision=float(0.0), recall=float(0.0), fmeasure=float(0.0))

    fmeasure = 2 * precision * recall / (precision + recall)
    return dict(
        precision=float(precision),
        recall=float(recall),
        fmeasure=float(fmeasure))


def _rougeL_score(pred: Sequence[str],
                  reference: Sequence[str]) -> Dict[str, float]:
    """计算Rouge-L metric的精度、召回率和F1得分值。

    :param pred: 一个预测句子的序列.
    :param reference: 一个标准译文句子的序列.
    """
    pred_len, reference_len = len(pred), len(reference)
    if 0 in (pred_len, reference_len):
        return dict(
            precision=float(0.0), recall=float(0.0), fmeasure=float(0.0))
    lcs = 0
    matches = SequenceMatcher(None, pred, reference).get_matching_blocks()
    for match in matches:
        lcs += match.size
    return _compute_metrics(lcs, pred_len, reference_len)


def _rougeN_score(pred: Sequence[str], reference: Sequence[str],
                  n_gram: int) -> Dict[str, float]:
    """计算Rouge-N metric的精度、召回率和F1得分值。

    :param pred: 一个预测句子的序列.
    :param reference: 一个标准译文句子的序列.
    :param n_gram: ``N-gram``值.
    """

    def get_n_gram(tokens: Sequence[str], n: int) -> Counter:
        ngrams: Counter = Counter()
        for ngram in (tuple(tokens[i:i + n])
                      for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams

    pred_ngarms = get_n_gram(pred, n_gram)
    reference_ngarms = get_n_gram(reference, n_gram)
    pred_len = sum(pred_ngarms.values())
    reference_len = sum(reference_ngarms.values())
    if 0 in (pred_len, reference_len):
        return dict(
            precision=float(0.0), recall=float(0.0), fmeasure=float(0.0))

    hits = sum(
        min(pred_ngarms[w], reference_ngarms[w]) for w in set(pred_ngarms))
    return _compute_metrics(hits, pred_len, reference_len)


class ROUGE(Metric):
    """计算ROUGE的Metric。

    :param rouge_keys (Union[List, Tuple,int,str]): 该参数包括要计算的各种类型的
        ROUGE的名称。包括 ``L`` 和 ``1`` 到 ``9``。默认值是 (1,2,``L``).
    :param use_stemmer: 使用Porter词干器去除单词后缀以提高匹配。
    :param normalizer: 用户自己的规范化函数。
        如果这值是``None``，则默认使用空格替换任何非字母数字字符。
        这个函数必须输入一个 ``str`` 并且返回 ``str``.
    :param tokenizer: 用户可以传入 Callable 函数进行分词。
        如果是``str``，则按照传入的语言进行分词，默认选项有['en','cn','zh']，``en``
        代表英语，其他代表中文。如果是 ``None``，则会在第一次 update 时选择第一个
        sample 的语言进行选择
    :param accumulate: 该参数在多 references 场景下使用。
        - ``avg`` 获取与预测相关的所有引用的平均值。
        - ``best`` 采用预测和多个对应参考之间获得的最佳fmmeasure得分。

    Examples:

        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here'] # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']] # noqa: E501
        >>> metric = ROUGE()
        >>> metric.update(predictions, references)
        >>> results = metric.get_metric()
    """

    def __init__(
        self,
        rouge_keys: Union[List, Tuple, int, str] = (1, 2, 'L'),
        use_stemmer: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Union[Callable, str]] = None,
        backend: Union[str, Backend, None] = 'auto',
        aggregate_when_get_metric: Optional[bool] = None,
        accumulate: str = 'best',
        **kwargs: Any,
    ):
        super().__init__(
            backend=backend,
            aggregate_when_get_metric=aggregate_when_get_metric)
        if isinstance(rouge_keys, int) or isinstance(rouge_keys, str):
            rouge_keys = [rouge_keys]
        for rouge_key in rouge_keys:
            if isinstance(rouge_key, int):
                if rouge_key < 1 or rouge_key > 9:
                    raise ValueError(
                        f'Got unknown rouge key {rouge_key}. Expected to be '
                        f'one of {1 - 9} or L')
            elif rouge_key != 'L':
                raise ValueError(
                    f'Got unknown rouge key {rouge_key}. Expected to be '
                    f'one of {1 - 9} or L')
        self.rouge_keys = rouge_keys
        if use_stemmer:
            if _module_available('nltk'):
                import nltk
                self.stemmer = nltk.stem.porter.PorterStemmer()
            else:
                raise ValueError(
                    'The nltk package is needed to use stemmer, '
                    'check https://www.nltk.org/install.html for installation.'
                )
        else:
            self.stemmer = None
        self.normalizer = normalizer

        if callable(tokenizer):
            # 检查是否一定是吃进去一个参数
            _check_valid_parameters_number(tokenizer, ['text'])
            self.tokenizer: Optional[Callable] = tokenizer
        elif isinstance(tokenizer, str):
            self.tokenizer = get_tokenizer(tokenizer)
            if self.tokenizer is None:
                raise ValueError('Right now, `tokenizer` only supports '
                                 "pre-defined 'en' or 'cn'.")
        else:
            assert tokenizer is None, \
                '`tokenizer_fn` supports Callable, str or None, ' \
                f'but not `{type(tokenizer)}`'
            self.tokenizer = tokenizer
        self.accumulate = accumulate
        self.register_element(
            name='total_samples',
            value=0,
            aggregate_method='sum',
            backend=backend)
        self.register_element(
            name='fmeasure',
            value=[0. for _ in range(len(rouge_keys))],
            aggregate_method='sum',
            backend=backend)
        self.register_element(
            name='precision',
            value=[0. for _ in range(len(rouge_keys))],
            aggregate_method='sum',
            backend=backend)
        self.register_element(
            name='recall',
            value=[0. for _ in range(len(rouge_keys))],
            aggregate_method='sum',
            backend=backend)

    def update(
        self, predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[str],
                          Sequence[Sequence[str]]]) -> None:
        r"""

        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。
        :param predictions: 预测的 ``sentence``，类型为``Sequence``，长度可变，假
            设为 ``L``。可以为 :class:`str` 类型，也可以为 :class:`list` 类型。
       :param references: 答案译文，type为``Sequence``，长度必须也为``L``，保持和
            ``predictions`` 一致，每一个元素也是一个``Sequence``。

           * references可以为str类型，但是该情况下predictions也必须为str类型。
           * references 可以为 :class:`list[str]` 类型，如果 predictions 只有一条
             数据，references数量不受限制；如果 predictions 数量超过一条，
             references 的长度必须匹配 predictions 的数量。
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

        if self.tokenizer is None:
            lang = 'en'
            for _char in predictions[0]:
                if '\u4e00' <= _char <= '\u9fa5':
                    lang = 'cn'
                    break
            self.tokenizer = get_tokenizer(lang)
        for prediction, _references in zip(predictions, references):
            self.total_samples += 1
            pred_token = _normalize_and_tokenize(prediction, self.tokenizer,
                                                 self.stemmer, self.normalizer)
            reference_len = len(_references)
            ref_tokens = []
            for reference in _references:
                ref_token = _normalize_and_tokenize(
                    reference,
                    self.tokenizer,
                    self.stemmer,
                    self.normalizer,
                )
                ref_tokens.append(ref_token)
            for i, rouge_key in enumerate(self.rouge_keys):
                fmeasure = precision = recall = 0.0
                for j, ref_token in enumerate(ref_tokens):
                    if isinstance(rouge_key, int):
                        score = _rougeN_score(pred_token, ref_token, rouge_key)
                    else:
                        score = _rougeL_score(pred_token, ref_token)
                    if self.accumulate == 'best':
                        if fmeasure < score['fmeasure']:
                            fmeasure = score['fmeasure']
                            precision = score['precision']
                            recall = score['recall']
                    else:
                        fmeasure += score['fmeasure']
                        precision += score['precision']
                        recall += score['recall']
                if self.accumulate == 'avg':
                    fmeasure, precision, recall = \
                        fmeasure / reference_len, \
                        precision / reference_len, recall / reference_len

                self.fmeasure[i] += fmeasure
                self.precision[i] += precision
                self.recall[i] += recall

    def get_metric(self) -> dict:
        r"""
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计
        算最终的评价结果。

        :return: 包含以下内容的字典：``{ 'rouge{x}_fmeasure': float,
            'rouge{x}_precision': float,
            'rouge{x}_recall': float}``，其中 {x} 的值取决于在初始化时传入的
            rouge_keys。
        """
        fmeasure, precision, recall = \
            self.fmeasure.to_list(), \
            self.precision.to_list(), self.recall.to_list()
        total_samples = self.total_samples.get_scalar()
        results = {}
        for i, rouge_key in enumerate(self.rouge_keys):
            results[f'rouge{rouge_key}_fmeasure'] = fmeasure[i] / total_samples
            results[
                f'rouge{rouge_key}_precision'] = precision[i] / total_samples
            results[f'rouge{rouge_key}_recall'] = recall[i] / total_samples
        return results

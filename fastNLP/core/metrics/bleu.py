__all__ = ['BLEU']

import re
from collections import Counter
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.metric import Metric
from fastNLP.core.utils.utils import _check_valid_parameters_number
from fastNLP.envs import _module_available

CHINESE_UCODE_RANGES = (
    ('\u3400', '\u4db5'),
    ('\u4e00', '\u9fa5'),
    ('\u9fa6', '\u9fbb'),
    ('\uf900', '\ufa2d'),
    ('\ufa30', '\ufa6a'),
    ('\ufa70', '\ufad9'),
    ('\u20000', '\u2a6d6'),
    ('\u2f800', '\u2fa1d'),
    ('\uff00', '\uffef'),
    ('\u2e80', '\u2eff'),
    ('\u3000', '\u303f'),
    ('\u31c0', '\u31ef'),
    ('\u2f00', '\u2fdf'),
    ('\u2ff0', '\u2fff'),
    ('\u3100', '\u312f'),
    ('\u31a0', '\u31bf'),
    ('\ufe10', '\ufe1f'),
    ('\ufe30', '\ufe4f'),
    ('\u2600', '\u26ff'),
    ('\u2700', '\u27bf'),
    ('\u3200', '\u32ff'),
    ('\u3300', '\u33ff'),
)

_REGEX = (
    # 对于以下特殊符号，在前面插入一个空格
    (re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
    # 除非前面有数字，否则句点或者逗号前后都加入一个空格
    (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
    # 除非后面有数字，否则句点或者逗号前后都加入一个空格
    (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
    # 前面是数字时，后面跟着的破折号前后各加一个空格
    (re.compile(r'([0-9])(-)'), r'\1 \2 '),
)

if _module_available('regex'):
    import regex

    _INT_REGEX = (
        # p{S} : 匹配所有的特殊符号，包括符号，运算符，标点符号等。
        # p{P} : 匹配所有的标点字符，包括符号，标点符号等，主要用于分隔单词
        # P{N} : 匹配所有非数字字符
        # 分离出前跟非数字的标点符号
        (regex.compile(r'(\P{N})(\p{P})'), r'\1 \2 '),
        # 分离出后跟非数字的标点符号
        (regex.compile(r'(\p{P})(\P{N})'), r' \1 \2'),
        # 分离出所有特殊符号
        (regex.compile(r'(\p{S})'), r' \1 '),
    )


class _BaseTokenizer:
    """sacre_bleu 中 tokenzier 的基类 。

    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(self, lowercase: bool = False) -> None:
        self.lowercase = lowercase

    @classmethod
    def _tokenize_regex(cls, line: str) -> str:
        r"""
       ``13a`` 和 ``zh`` 的通用后处理 tokenizer。

       :param line: 要标记的的输入。
       :return: 被正则化后的输入。
       """
        for (_re, repl) in _REGEX:
            line = _re.sub(repl, line)
        return ' '.join(line.split())

    @classmethod
    def _tokenize_base(cls, line: str) -> str:
        r"""
       使用 tokenizer 对于输入行进行分词。

       :param line: 要标记的的输入。
       :return: 返回处理过的输入。
       """
        return line

    @staticmethod
    def _lower(line: str, lowercase: bool) -> str:
        r"""
       使用 tokenizer 对于输入行进行分词。

       :param line: 要标记的的输入。
       :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
       :return: 返回处理过的输入。
       """
        if lowercase:
            return line.lower()
        return line

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_base(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _13aTokenizer(_BaseTokenizer):
    """sacre_bleu 中实现 ``13a`` 的 tokenizer 。

    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_13a(cls, line: str) -> str:
        r"""
       使用相对最小的 tokenizer 对输入行进行标记化，参照 mteval-v13a，WMT 使用。

       :param line: 要标记的的输入。
       :return: 返回处理过的输入。
       """
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')

        if '&' in line:
            line = line.replace('&quot;', '"')
            line = line.replace('&amp;', '&')
            line = line.replace('&lt;', '<')
            line = line.replace('&gt;', '>')

        return cls._tokenize_regex(line)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_13a(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _zhTokenizer(_BaseTokenizer):
    """sacre_bleu 中实现 ``zh`` 的 tokenizer 。

    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @staticmethod
    def _is_chinese_char(uchar: str) -> bool:
        r"""
       判断是否中文。

       :param uchar: unicode 中的输入字符。
       :return: 返回是否中文的bool值。
       """
        for start, end in CHINESE_UCODE_RANGES:
            if start <= uchar <= end:
                return True
        return False

    @classmethod
    def _tokenize_zh(cls, line: str) -> str:
        r"""
        使用相对最小的 tokenizer 对输入行进行标记化，参 照mteval-v13a，WMT 使用。
        其中，中文部分就是直接在中文汉字前后加入空格，其他还是按照 13a 中的标准进行分
        词。

       :param line: 要标记的的输入。
       :return: 返回处理过的输入。
       """
        line = line.strip()
        line_in_chars = ''

        for char in line:
            if cls._is_chinese_char(char):
                line_in_chars += ' '
                line_in_chars += char
                line_in_chars += ' '
            else:
                line_in_chars += char

        return cls._tokenize_regex(line_in_chars)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_zh(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _intlTokenizer(_BaseTokenizer):
    """sacre_bleu 中实现 ``international`` 的 tokenizer 。

    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_international(cls, line: str) -> str:
        r"""
       应用国际标记化并模仿 Moses 的脚本 mteval-v14

       :param line: 要标记的的输入。
       :return: 返回处理过的输入。
       """
        for (_re, repl) in _INT_REGEX:
            line = _re.sub(repl, line)

        return ' '.join(line.split())

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_international(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _charTokenizer(_BaseTokenizer):
    """sacre_bleu 中实现 ``char`` 的 tokenizer 。

    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_char(cls, line: str) -> str:
        r"""
       用于与语言无关的字符级标记化。

       :param line: 要标记的的输入。
       :return: 返回处理过的输入。
       """
        return ' '.join(char for char in line)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_char(line)
        return self._lower(tokenized_line, self.lowercase).split()


def get_n_gram(token: Sequence[str], n_gram: int) -> Counter:
    counter: Counter = Counter()
    for i in range(1, n_gram + 1):
        for j in range(len(token) - i + 1):
            key = tuple(token[j:(i + j)])
            counter[key] += 1
    return counter


def get_tokenizer(tokenizer: str, lowercase: bool):
    if tokenizer == 'none':
        tokenizer_fn = _BaseTokenizer(lowercase)
    elif tokenizer == '13a':
        tokenizer_fn = _13aTokenizer(lowercase)
    elif tokenizer == 'zh':
        tokenizer_fn = _zhTokenizer(lowercase)
    elif tokenizer == 'intl':
        tokenizer_fn = _intlTokenizer(lowercase)
    elif tokenizer == 'char':
        tokenizer_fn = _charTokenizer(lowercase)
    else:
        raise ValueError('Right now, `tokenizer_fn` only supports pre-defined '
                         "'none', '13a', 'intl', 'char', 'zh'.")
    return tokenizer_fn


def _get_brevity_penalty(pred_len: float, references_len: float) -> float:
    if pred_len >= references_len:
        return float(1.)
    elif pred_len == 0 or references_len == 0:
        return float(0.)
    return np.exp(1 - references_len / pred_len)


class BLEU(Metric):
    r"""计算 **bleu** 的 ``Metric`` 。

    :param n_gram: Gram 的范围是 ``[1, 4]``
    :param smooth: 是否选择 **smoothing** 计算
    :param ngram_weights: 用来控制各个 i-gram 所计算结果的权重，需要满足
        sum(nrgam_weights) 为 **1**。
    :param backend: 目前支持五种类型的backend, ``['auto', 'torch', 'paddle',
        'jittor', 'oneflow']``。其中 ``'auto'`` 表示根据实际调用。:meth:`update`
        函数时传入的参数决定具体的 backend ，一般情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相
        同的 element 的数字聚合后再得到 metric，当 ``backend`` 不支持分布式时，该参
        数无意义。如果为 ``None``，将在 :class:`.Evaluator` 中根据 ``sampler`` 是
        否使用分布式进行自动设置。
    :param tokenizer: 用户可以传入 Callable 函数进行分词。如果是 ``str``，则按照传
        入的 str 选择对应的 tokenizer，默认选项有 ``['none','13a','zh', 'char',
        'intl']``：

        * 值为 ``"none"`` 时，
          tokenizer 将只进行分词处理，不进行其他处理。
        * 值为 ``"13a"`` 时，
          将对于句子中的特殊符号，标点数字进行正则化再进行分词，模仿的是 ``Moses``
          的 ``mteval-v13a``。
        * 值为 ``"zh"`` 时，
          输入的数据为中文句子，tokenizer 为在每一个汉字前后加入空格，以做切分。
        * 值为 ``"char"`` 时，
          tokenizer 会在输入句子的每一个 char 前后加入空格，以 char 为单位进行切
          分。
        * 值为 ``"intl"`` 时，
          使用国际化进行词语切分，规则仿照于 ``13a``。
          如果是 ``Callable``，则选择用户自定义的 tokenizer。
          如果是 ``None``，则会再第一次 update 时选择第一个 sample 的语言进行选择。
    :param lowercase: 在设定 tokenizer 时是否将字母小写处理。
    """

    def __init__(
        self,
        n_gram: int = 4,
        smooth: bool = False,
        ngram_weights: Optional[Sequence[float]] = None,
        backend: Union[str, Backend, None] = 'auto',
        aggregate_when_get_metric: Optional[bool] = None,
        tokenizer_fn: Optional[Union[Callable, str]] = None,
        lowercase: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            backend=backend,
            aggregate_when_get_metric=aggregate_when_get_metric)
        self.n_gram = n_gram
        self.smooth = smooth
        if ngram_weights is not None and len(ngram_weights) != n_gram:
            raise ValueError(
                'The number of weights in weights is different from n_gram: '
                f'{len(ngram_weights)} != {n_gram}')
        if ngram_weights is not None:
            self.ngram_weights = ngram_weights
        else:
            self.ngram_weights = [1.0 / n_gram] * n_gram

        self.register_element(
            name='pred_len', value=0, aggregate_method='sum', backend=backend)
        self.register_element(
            name='references_len',
            value=0,
            aggregate_method='sum',
            backend=backend)
        self.register_element(
            name='precision_matches',
            value=[0 for _ in range(self.n_gram)],
            aggregate_method='sum',
            backend=backend)
        self.register_element(
            name='precision_total',
            value=[0 for _ in range(self.n_gram)],
            aggregate_method='sum',
            backend=backend)

        if tokenizer_fn is None:
            tokenizer_fn = 'none'
        if tokenizer_fn == 'intl' and not _module_available('regex'):
            raise ValueError(
                '`intl` needs regex package, please make sure you have '
                'already installed it`')
        if callable(tokenizer_fn):
            # 检查是否一定是吃进去一个参数
            _check_valid_parameters_number(tokenizer_fn, ['text'])
            self.tokenizer_fn: Optional[Callable] = tokenizer_fn
        elif isinstance(tokenizer_fn, str):
            self.tokenizer_fn = get_tokenizer(tokenizer_fn, lowercase)
        else:
            raise ValueError('`tokenizer_fn` supports Callable, str or None, '
                             f'but not `{type(tokenizer_fn)}`')

    def update(
        self, predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[str],
                          Sequence[Sequence[str]]]) -> None:
        r"""
        :meth:`update` 函数将针对一个批次的预测结果做评价指标的累计。

        :param predictions: 预测的 ``sentence``，类型为 ``Sequence``，长度可变，
            假设为 ``L``。可以为 :class:`str` 类型，也可以为 :class:`list` 类型。
        :param references: 答案译文，类型为 ``Sequence``，长度必须也为 ``L``，
            保持和 ``predictions`` 一致，每一个元素也是一个 ``Sequence``。

            * references 可以为 :class:`str` 类型，但是该情况下 predictions 也必
              须为 :class:`str` 类型。
            * references 可以为 :class:`list[str]` 类型，如果 predictions 只有一
              条数据，references 数量不受限制；如果 predictions 数量超过一条，
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

        references_token: Sequence[Sequence[Sequence[str]]] = [
            [
                self.tokenizer_fn(line)  # type: ignore
                for line in r
            ] for r in references
        ]
        predictions_token: Sequence[Sequence[str]] = [
            self.tokenizer_fn(line) for line in predictions  # type: ignore
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
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计
        算最终的评价结果。

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

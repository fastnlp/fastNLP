__all__ = [
    "CWSPipe"
]

import re
from itertools import chain

from .pipe import Pipe
from .utils import _indexize
from fastNLP.io.data_bundle import DataBundle
from fastNLP.io.loader import CWSLoader
# from ...core.const import Const


def _word_lens_to_bmes(word_lens):
    r"""

    :param list word_lens: List[int], 每个词语的长度
    :return: List[str], BMES的序列
    """
    tags = []
    for word_len in word_lens:
        if word_len == 1:
            tags.append('S')
        else:
            tags.append('B')
            tags.extend(['M'] * (word_len - 2))
            tags.append('E')
    return tags


def _word_lens_to_segapp(word_lens):
    r"""

    :param list word_lens: List[int], 每个词语的长度
    :return: List[str], BMES的序列
    """
    tags = []
    for word_len in word_lens:
        if word_len == 1:
            tags.append('SEG')
        else:
            tags.extend(['APP'] * (word_len - 1))
            tags.append('SEG')
    return tags


def _alpha_span_to_special_tag(span):
    r"""
    将span替换成特殊的字符

    :param str span:
    :return:
    """
    if 'oo' == span.lower():  # speical case when represent 2OO8
        return span
    if len(span) == 1:
        return span
    else:
        return '<ENG>'


def _find_and_replace_alpha_spans(line):
    r"""
    传入原始句子，替换其中的字母为特殊标记

    :param str line:原始数据
    :return: str
    """
    new_line = ''
    pattern = '[a-zA-Z]+(?=[\u4e00-\u9fff ，％,.。！<－“])'
    prev_end = 0
    for match in re.finditer(pattern, line):
        start, end = match.span()
        span = line[start:end]
        new_line += line[prev_end:start] + _alpha_span_to_special_tag(span)
        prev_end = end
    new_line += line[prev_end:]
    return new_line


def _digit_span_to_special_tag(span):
    r"""

    :param str span: 需要替换的str
    :return:
    """
    if span[0] == '0' and len(span) > 2:
        return '<NUM>'
    decimal_point_count = 0  # one might have more than one decimal pointers
    for idx, char in enumerate(span):
        if char == '.' or char == '﹒' or char == '·':
            decimal_point_count += 1
    if span[-1] == '.' or span[-1] == '﹒' or span[
        -1] == '·':  # last digit being decimal point means this is not a number
        if decimal_point_count == 1:
            return span
        else:
            return '<UNKDGT>'
    if decimal_point_count == 1:
        return '<DEC>'
    elif decimal_point_count > 1:
        return '<UNKDGT>'
    else:
        return '<NUM>'


def _find_and_replace_digit_spans(line):
    r"""
    only consider words start with number, contains '.', characters.
    
        If ends with space, will be processed
        
        If ends with Chinese character, will be processed
        
        If ends with or contains english char, not handled.
    
    floats are replaced by <DEC>
    
    otherwise unkdgt
    """
    new_line = ''
    pattern = r'\d[\d\\.﹒·]*(?=[\u4e00-\u9fff  ，％%,。！<－“])'
    prev_end = 0
    for match in re.finditer(pattern, line):
        start, end = match.span()
        span = line[start:end]
        new_line += line[prev_end:start] + _digit_span_to_special_tag(span)
        prev_end = end
    new_line += line[prev_end:]
    return new_line


class CWSPipe(Pipe):
    r"""
    对 **CWS** 数据进行处理，处理之后 :class:`~fastNLP.core.DataSet` 中的内容如下：

    .. csv-table::
       :header: "raw_words", "chars", "target", "seq_len"

       "共同  创造  美好...", "[2, 3, 4...]", "[0, 2, 0, 2,...]", 13
       "2001年  新年  钟声...", "[8, 9, 9, 7, ...]", "[0, 1, 1, 1, 2...]", 20
       "...", "[...]","[...]", .

    :param dataset_name: data 的名称，支持 ``['pku', 'msra', 'cityu'(繁体), 'as'(繁体), None]``
    :param encoding_type: ``target`` 列使用什么类型的 encoding 方式，支持 ``['bmes', 'segapp']`` 两种。``"我 来自 复旦大学..."`` 这句话 ``bmes``的 
        tag为 ``[S, B, E, B, M, M, E...]`` ； ``segapp`` 的 tag 为 ``[seg, app, seg, app, app, app, seg, ...]`` 。
    :param replace_num_alpha: 是否将数字和字母用特殊字符替换。
    :param bigrams: 是否增加一列 ``bigrams`` 。 ``bigrams`` 会对原文进行如下转化： ``['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]`` 。如果
        设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``bigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('bigrams')`` 获取。
    :param trigrams: 是否增加一列 ``trigrams`` 。 ``trigrams`` 会对原文进行如下转化 ``['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]`` 。
        如果设置为 ``True`` ，返回的 :class:`~fastNLP.core.DataSet` 将有一列名为 ``trigrams`` ，且已经转换为了 index 并设置为 input，对应的词表可以通过
        ``data_bundle.get_vocab('trigrams')`` 获取。
    :param num_proc: 处理数据时使用的进程数目。
    """
    
    def __init__(self, dataset_name: str=None, encoding_type: str='bmes', replace_num_alpha: bool=True,
                 bigrams: bool=False, trigrams: bool=False, num_proc: int = 0):
        if encoding_type == 'bmes':
            self.word_lens_to_tags = _word_lens_to_bmes
        else:
            self.word_lens_to_tags = _word_lens_to_segapp
        
        self.dataset_name = dataset_name
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.replace_num_alpha = replace_num_alpha
        self.num_proc = num_proc
    
    def _tokenize(self, data_bundle):
        r"""
        将data_bundle中的'chars'列切分成一个一个的word.
        例如输入是"共同  创造  美好.."->[[共, 同], [创, 造], [...], ]

        :param data_bundle:
        :return:
        """
        def split_word_into_chars(raw_chars):
            words = raw_chars.split()
            chars = []
            for word in words:
                char = []
                subchar = []
                for c in word:
                    if c == '<':
                        if subchar:
                            char.extend(subchar)
                            subchar = []
                        subchar.append(c)
                        continue
                    if c == '>' and len(subchar)>0 and subchar[0] == '<':
                        subchar.append(c)
                        char.append(''.join(subchar))
                        subchar = []
                        continue
                    if subchar:
                        subchar.append(c)
                    else:
                        char.append(c)
                char.extend(subchar)
                chars.append(char)
            return chars
        
        for name, dataset in data_bundle.iter_datasets():
            dataset.apply_field(split_word_into_chars, field_name='chars',
                                new_field_name='chars', num_proc=self.num_proc)
        return data_bundle
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        ``data_bunlde`` 中的 :class:`~fastNLP.core.DataSet` 应该包含 ``raw_words`` ：

        .. csv-table::
           :header: "raw_words"

           "上海 浦东 开发 与 法制 建设 同步"
           "新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）"
           "..."

        :param data_bundle:
        :return: 处理后的 ``data_bundle``
        """
        data_bundle.copy_field('raw_words', 'chars')
        
        if self.replace_num_alpha:
            data_bundle.apply_field(_find_and_replace_alpha_spans, 'chars', 'chars', num_proc=self.num_proc)
            data_bundle.apply_field(_find_and_replace_digit_spans, 'chars', 'chars', num_proc=self.num_proc)
        
        self._tokenize(data_bundle)

        def func1(chars):
            return self.word_lens_to_tags(map(len, chars))

        def func2(chars):
            return list(chain(*chars))
        
        for name, dataset in data_bundle.iter_datasets():
            dataset.apply_field(func1, field_name='chars', new_field_name='target', num_proc=self.num_proc)
            dataset.apply_field(func2, field_name='chars', new_field_name='chars', num_proc=self.num_proc)
        input_field_names = ['chars']

        def bigram(chars):
            return [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])]

        def trigrams(chars):
            return [c1 + c2 + c3 for c1, c2, c3 in
                    zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)]

        if self.bigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(bigram, field_name='chars', new_field_name='bigrams', num_proc=self.num_proc)
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.iter_datasets():
                dataset.apply_field(trigrams, field_name='chars', new_field_name='trigrams', num_proc=self.num_proc)
            input_field_names.append('trigrams')
        
        _indexize(data_bundle, input_field_names, 'target')

        for name, dataset in data_bundle.iter_datasets():
            dataset.add_seq_len('chars')

        return data_bundle
    
    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        if self.dataset_name is None and paths is None:
            raise RuntimeError(
                "You have to set `paths` when calling process_from_file() or `dataset_name `when initialization.")
        if self.dataset_name is not None and paths is not None:
            raise RuntimeError("You cannot specify `paths` and `dataset_name` simultaneously")
        data_bundle = CWSLoader(self.dataset_name).load(paths)
        return self.process(data_bundle)

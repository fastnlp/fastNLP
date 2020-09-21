r"""undocumented"""

__all__ = [
    "CWSPipe"
]

import re
from itertools import chain

from .pipe import Pipe
from .utils import _indexize
from .. import DataBundle
from ..loader import CWSLoader
from ...core.const import Const


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
    pattern = '\d[\d\\.﹒·]*(?=[\u4e00-\u9fff  ，％%,。！<－“])'
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
    对CWS数据进行预处理, 处理之后的数据，具备以下的结构

    .. csv-table::
       :header: "raw_words", "chars", "target", "seq_len"

       "共同  创造  美好...", "[2, 3, 4...]", "[0, 2, 0, 2,...]", 13
       "2001年  新年  钟声...", "[8, 9, 9, 7, ...]", "[0, 1, 1, 1, 2...]", 20
       "...", "[...]","[...]", .

    dataset的print_field_meta()函数输出的各个field的被设置成input和target的情况为::

        +-------------+-----------+-------+--------+---------+
        | field_names | raw_words | chars | target | seq_len |
        +-------------+-----------+-------+--------+---------+
        |   is_input  |   False   |  True |  True  |   True  |
        |  is_target  |   False   | False |  True  |   True  |
        | ignore_type |           | False | False  |  False  |
        |  pad_value  |           |   0   |   0    |    0    |
        +-------------+-----------+-------+--------+---------+

    """
    
    def __init__(self, dataset_name=None, encoding_type='bmes', replace_num_alpha=True, bigrams=False, trigrams=False):
        r"""
        
        :param str,None dataset_name: 支持'pku', 'msra', 'cityu', 'as', None
        :param str encoding_type: 可以选择'bmes', 'segapp'两种。"我 来自 复旦大学...", bmes的tag为[S, B, E, B, M, M, E...]; segapp
            的tag为[seg, app, seg, app, app, app, seg, ...]
        :param bool replace_num_alpha: 是否将数字和字母用特殊字符替换。
        :param bool bigrams: 是否增加一列bigram. bigram的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]
        :param bool trigrams: 是否增加一列trigram. trigram的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
        """
        if encoding_type == 'bmes':
            self.word_lens_to_tags = _word_lens_to_bmes
        else:
            self.word_lens_to_tags = _word_lens_to_segapp
        
        self.dataset_name = dataset_name
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.replace_num_alpha = replace_num_alpha
    
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
        
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(split_word_into_chars, field_name=Const.CHAR_INPUT,
                                new_field_name=Const.CHAR_INPUT)
        return data_bundle
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        可以处理的DataSet需要包含raw_words列

        .. csv-table::
           :header: "raw_words"

           "上海 浦东 开发 与 法制 建设 同步"
           "新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）"
           "..."

        :param data_bundle:
        :return:
        """
        data_bundle.copy_field(Const.RAW_WORD, Const.CHAR_INPUT)
        
        if self.replace_num_alpha:
            data_bundle.apply_field(_find_and_replace_alpha_spans, Const.CHAR_INPUT, Const.CHAR_INPUT)
            data_bundle.apply_field(_find_and_replace_digit_spans, Const.CHAR_INPUT, Const.CHAR_INPUT)
        
        self._tokenize(data_bundle)
        
        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(lambda chars: self.word_lens_to_tags(map(len, chars)), field_name=Const.CHAR_INPUT,
                                new_field_name=Const.TARGET)
            dataset.apply_field(lambda chars: list(chain(*chars)), field_name=Const.CHAR_INPUT,
                                new_field_name=Const.CHAR_INPUT)
        input_field_names = [Const.CHAR_INPUT]
        if self.bigrams:
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(lambda chars: [c1 + c2 for c1, c2 in zip(chars, chars[1:] + ['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')
        if self.trigrams:
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(lambda chars: [c1 + c2 + c3 for c1, c2, c3 in
                                                   zip(chars, chars[1:] + ['<eos>'], chars[2:] + ['<eos>'] * 2)],
                                    field_name=Const.CHAR_INPUT, new_field_name='trigrams')
            input_field_names.append('trigrams')
        
        _indexize(data_bundle, input_field_names, Const.TARGET)
        
        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET, Const.INPUT_LEN]
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)
        
        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)
        
        return data_bundle
    
    def process_from_file(self, paths=None) -> DataBundle:
        r"""
        
        :param str paths:
        :return:
        """
        if self.dataset_name is None and paths is None:
            raise RuntimeError(
                "You have to set `paths` when calling process_from_file() or `dataset_name `when initialization.")
        if self.dataset_name is not None and paths is not None:
            raise RuntimeError("You cannot specify `paths` and `dataset_name` simultaneously")
        data_bundle = CWSLoader(self.dataset_name).load(paths)
        return self.process(data_bundle)

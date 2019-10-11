from fastNLP.io.pipe import Pipe
from fastNLP.io import DataBundle
from fastNLP.io.loader import CWSLoader
from fastNLP import Const
from itertools import chain
from fastNLP.io.pipe.utils import _indexize
from functools import partial
from fastNLP.io.pipe.cws import _find_and_replace_alpha_spans, _find_and_replace_digit_spans


def _word_lens_to_relay(word_lens):
    """
    [1, 2, 3, ..] 转换为[0, 1, 0, 2, 1, 0,](start指示seg有多长);
    :param word_lens:
    :return:
    """
    tags = []
    for word_len in word_lens:
        tags.extend([idx for idx in range(word_len - 1, -1, -1)])
    return tags

def _word_lens_to_end_seg_mask(word_lens):
    """
    [1, 2, 3, ..] 转换为[0, 1, 0, 2, 1, 0,](start指示seg有多长);
    :param word_lens:
    :return:
    """
    end_seg_mask = []
    for word_len in word_lens:
        end_seg_mask.extend([0] * (word_len - 1) + [1])
    return end_seg_mask

def _word_lens_to_start_seg_mask(word_lens):
    """
    [1, 2, 3, ..] 转换为[0, 1, 0, 2, 1, 0,](start指示seg有多长);
    :param word_lens:
    :return:
    """
    start_seg_mask = []
    for word_len in word_lens:
        start_seg_mask.extend([1] + [0] * (word_len - 1))
    return start_seg_mask


class CWSShiftRelayPipe(Pipe):
    """

    :param str,None dataset_name: 支持'pku', 'msra', 'cityu', 'as', None
    :param int L: ShiftRelay模型的超参数
    :param bool replace_num_alpha: 是否将数字和字母用特殊字符替换。
    :param bool bigrams: 是否增加一列bigram. bigram的构成是['复', '旦', '大', '学', ...]->["复旦", "旦大", ...]
    :param bool trigrams: 是否增加一列trigram. trigram的构成是 ['复', '旦', '大', '学', ...]->["复旦大", "旦大学", ...]
    """
    def __init__(self, dataset_name=None, L=5, replace_num_alpha=True, bigrams=True):
        self.dataset_name = dataset_name
        self.bigrams = bigrams
        self.replace_num_alpha = replace_num_alpha
        self.L = L

    def _tokenize(self, data_bundle):
        """
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
                    if c=='<':
                        subchar.append(c)
                        continue
                    if c=='>' and subchar[0]=='<':
                        char.append(''.join(subchar))
                        subchar = []
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
        """
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
        input_field_names = [Const.CHAR_INPUT]
        target_field_names = []

        for name, dataset in data_bundle.datasets.items():
            dataset.apply_field(lambda chars:_word_lens_to_relay(map(len, chars)), field_name=Const.CHAR_INPUT,
                                new_field_name=Const.TARGET)
            dataset.apply_field(lambda chars:_word_lens_to_start_seg_mask(map(len, chars)), field_name=Const.CHAR_INPUT,
                                new_field_name='start_seg_mask')
            dataset.apply_field(lambda chars:_word_lens_to_end_seg_mask(map(len, chars)), field_name=Const.CHAR_INPUT,
                                new_field_name='end_seg_mask')
            dataset.apply_field(lambda chars:list(chain(*chars)), field_name=Const.CHAR_INPUT,
                                new_field_name=Const.CHAR_INPUT)
            target_field_names.append('start_seg_mask')
            input_field_names.append('end_seg_mask')
        if self.bigrams:
            for name, dataset in data_bundle.datasets.items():
                dataset.apply_field(lambda chars: [c1+c2 for c1, c2 in zip(chars, chars[1:]+['<eos>'])],
                                    field_name=Const.CHAR_INPUT, new_field_name='bigrams')
            input_field_names.append('bigrams')

        _indexize(data_bundle, ['chars', 'bigrams'], [])

        func = partial(_clip_target, L=self.L)
        for name, dataset in data_bundle.datasets.items():
            res = dataset.apply_field(func, field_name='target')
            relay_target = [res_i[0] for res_i in res]
            relay_mask = [res_i[1] for res_i in res]
            dataset.add_field('relay_target', relay_target, is_input=True, is_target=False, ignore_type=False)
            dataset.add_field('relay_mask', relay_mask, is_input=True, is_target=False, ignore_type=False)
            input_field_names.append('relay_target')
            input_field_names.append('relay_mask')

        input_fields = [Const.TARGET, Const.INPUT_LEN] + input_field_names
        target_fields = [Const.TARGET, Const.INPUT_LEN] + target_field_names
        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)

        data_bundle.set_input(*input_fields)
        data_bundle.set_target(*target_fields)

        return data_bundle

    def process_from_file(self, paths=None) -> DataBundle:
        """

        :param str paths:
        :return:
        """
        if self.dataset_name is None and paths is None:
            raise RuntimeError("You have to set `paths` when calling process_from_file() or `dataset_name `when initialization.")
        if self.dataset_name is not None and paths is not None:
            raise RuntimeError("You cannot specify `paths` and `dataset_name` simultaneously")
        data_bundle = CWSLoader(self.dataset_name).load(paths)
        return self.process(data_bundle)

def _clip_target(target, L:int):
    """

    只有在target_type为shift_relay的使用
    :param target: List[int]
    :param L:
    :return:
    """
    relay_target_i = []
    tmp = []
    for j in range(len(target) - 1):
        tmp.append(target[j])
        if target[j] > target[j + 1]:
            pass
        else:
            relay_target_i.extend([L - 1 if t >= L else t for t in tmp[::-1]])
            tmp = []
    # 处理未结束的部分
    if len(tmp) == 0:
        relay_target_i.append(0)
    else:
        tmp.append(target[-1])
        relay_target_i.extend([L - 1 if t >= L else t for t in tmp[::-1]])
    relay_mask_i = []
    j = 0
    while j < len(target):
        seg_len = target[j] + 1
        if target[j] < L:
            relay_mask_i.extend([0] * (seg_len))
        else:
            relay_mask_i.extend([1] * (seg_len - L) + [0] * L)
        j = seg_len + j
    return relay_target_i, relay_mask_i

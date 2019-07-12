
import re

from fastNLP.api.processor import Processor
from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
from reproduction.legacy.Chinese_word_segmentation.process.span_converter import SpanConverter

_SPECIAL_TAG_PATTERN = '<[a-zA-Z]+>'

class SpeicalSpanProcessor(Processor):
    """
    将DataSet中field_name使用span_converter替换掉。

    """
    def __init__(self, field_name, new_added_field_name=None):
        super(SpeicalSpanProcessor, self).__init__(field_name, new_added_field_name)

        self.span_converters = []


    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        def inner_proc(ins):
            sentence = ins[self.field_name]
            for span_converter in self.span_converters:
                sentence = span_converter.find_certain_span_and_replace(sentence)
            return sentence
        dataset.apply(func=inner_proc, new_field_name=self.new_added_field_name)

        return dataset

    def add_span_converter(self, converter):
        assert isinstance(converter, SpanConverter), "Only SpanConverterBase is allowed, not {}."\
            .format(type(converter))
        self.span_converters.append(converter)


class CWSCharSegProcessor(Processor):
    """
    将DataSet中field_name这个field分成一个个的汉字，即原来可能为"复旦大学 fudan", 分成['复', '旦', '大', '学',
        ' ', 'f', 'u', ...]

    """
    def __init__(self, field_name, new_added_field_name):
        super(CWSCharSegProcessor, self).__init__(field_name, new_added_field_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        def inner_proc(ins):
            sentence = ins[self.field_name]
            chars = self._split_sent_into_chars(sentence)
            return chars
        dataset.apply(func=inner_proc, new_field_name=self.new_added_field_name)

        return dataset

    def _split_sent_into_chars(self, sentence):
        sp_tag_match_iter = re.finditer(_SPECIAL_TAG_PATTERN, sentence)
        sp_spans = [match_span.span() for match_span in sp_tag_match_iter]
        sp_span_idx = 0
        in_span_flag = False
        chars = []
        num_spans = len(sp_spans)
        for idx, char in enumerate(sentence):
            if sp_span_idx<num_spans and idx == sp_spans[sp_span_idx][0]:
                in_span_flag = True
            elif in_span_flag and sp_span_idx<num_spans and idx == sp_spans[sp_span_idx][1] - 1:
                chars.append(sentence[sp_spans[sp_span_idx]
                                      [0]:sp_spans[sp_span_idx][1]])
                in_span_flag = False
                sp_span_idx += 1
            elif not in_span_flag:
                # TODO 需要谨慎考虑如何处理空格的问题
                if char != ' ':
                    chars.append(char)
            else:
                pass
        return chars


class CWSTagProcessor(Processor):
    """
    为分词生成tag。该class为Base class。

    """
    def __init__(self, field_name, new_added_field_name=None):
        super(CWSTagProcessor, self).__init__(field_name, new_added_field_name)

    def _generate_tag(self, sentence):
        sp_tag_match_iter = re.finditer(_SPECIAL_TAG_PATTERN, sentence)
        sp_spans = [match_span.span() for match_span in sp_tag_match_iter]
        sp_span_idx = 0
        in_span_flag = False
        tag_list = []
        word_len = 0
        num_spans = len(sp_spans)
        for idx, char in enumerate(sentence):
            if sp_span_idx<num_spans and idx == sp_spans[sp_span_idx][0]:
                in_span_flag = True
            elif in_span_flag and sp_span_idx<num_spans and idx == sp_spans[sp_span_idx][1] - 1:
                word_len += 1
                in_span_flag = False
                sp_span_idx += 1
            elif not in_span_flag:
                if char == ' ':
                    if word_len!=0:
                        tag_list.extend(self._tags_from_word_len(word_len))
                    word_len = 0
                else:
                    word_len += 1
            else:
                pass
        if word_len!=0:
            tag_list.extend(self._tags_from_word_len(word_len))

        return tag_list

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        def inner_proc(ins):
            sentence = ins[self.field_name]
            tag_list = self._generate_tag(sentence)
            return tag_list
        dataset.apply(func=inner_proc, new_field_name=self.new_added_field_name)
        dataset.set_target(self.new_added_field_name)
        return dataset

    def _tags_from_word_len(self, word_len):
        raise NotImplementedError

class CWSBMESTagProcessor(CWSTagProcessor):
    """
    通过DataSet中的field_name这个field生成相应的BMES的tag。

    """
    def __init__(self, field_name, new_added_field_name=None):
        super(CWSBMESTagProcessor, self).__init__(field_name, new_added_field_name)

        self.tag_size = 4

    def _tags_from_word_len(self, word_len):
        tag_list = []
        if word_len == 1:
            tag_list.append(3)
        else:
            tag_list.append(0)
            for _ in range(word_len-2):
                tag_list.append(1)
            tag_list.append(2)

        return tag_list

class CWSSegAppTagProcessor(CWSTagProcessor):
    """
    通过DataSet中的field_name这个field生成相应的SegApp的tag。

    """
    def __init__(self, field_name, new_added_field_name=None):
        super(CWSSegAppTagProcessor, self).__init__(field_name, new_added_field_name)

        self.tag_size = 2

    def _tags_from_word_len(self, word_len):
        tag_list = []
        for _ in range(word_len-1):
            tag_list.append(0)
        tag_list.append(1)
        return tag_list


class BigramProcessor(Processor):
    """
    这是生成bigram的基类。

    """
    def __init__(self, field_name, new_added_fielf_name=None):

        super(BigramProcessor, self).__init__(field_name, new_added_fielf_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))

        def inner_proc(ins):
            characters = ins[self.field_name]
            bigrams = self._generate_bigram(characters)
            return bigrams
        dataset.apply(func=inner_proc, new_field_name=self.new_added_field_name)

        return dataset

    def _generate_bigram(self, characters):
        pass


class Pre2Post2BigramProcessor(BigramProcessor):
    """
    该bigram processor生成bigram的方式如下
    原汉字list为l = ['a', 'b', 'c']，会被padding为L=['SOS', 'SOS', 'a', 'b', 'c', 'EOS', 'EOS']，生成bigram list为
        [L[idx-2], L[idx-1], L[idx+1], L[idx+2], L[idx-2]L[idx], L[idx-1]L[idx], L[idx]L[idx+1], L[idx]L[idx+2], ....]
    即每个汉字，会有八个bigram, 对于上例中'a'的bigram为
        ['SOS', 'SOS', 'b', 'c', 'SOSa', 'SOSa', 'ab', 'ac']
    返回的bigram是一个list，但其实每8个元素是一个汉字的bigram信息。

    """
    def __init__(self, field_name, new_added_field_name=None):

        super(BigramProcessor, self).__init__(field_name, new_added_field_name)

    def _generate_bigram(self, characters):
        bigrams = []
        characters = ['<SOS>', '<SOS>'] + characters + ['<EOS>', '<EOS>']
        for idx in range(2, len(characters)-2):
            cur_char = characters[idx]
            pre_pre_char = characters[idx-2]
            pre_char = characters[idx-1]
            post_char = characters[idx+1]
            post_post_char = characters[idx+2]
            pre_pre_cur_bigram = pre_pre_char + cur_char
            pre_cur_bigram = pre_char + cur_char
            cur_post_bigram = cur_char + post_char
            cur_post_post_bigram = cur_char + post_post_char
            bigrams.extend([pre_pre_char, pre_char, post_char, post_post_char,
                            pre_pre_cur_bigram, pre_cur_bigram,
                            cur_post_bigram, cur_post_post_bigram])
        return bigrams


class VocabProcessor(Processor):
    def __init__(self, field_name, min_freq=1, max_size=None):

        super(VocabProcessor, self).__init__(field_name, None)
        self.vocab = Vocabulary(min_freq=min_freq, max_size=max_size)

    def process(self, *datasets):
        for dataset in datasets:
            assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
            dataset.apply(lambda ins: self.vocab.update(ins[self.field_name]))

    def get_vocab(self):
        self.vocab.build_vocab()
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)


class SegApp2OutputProcessor(Processor):
    def __init__(self, chars_field_name='chars_list', tag_field_name='pred_tags', new_added_field_name='output'):
        super(SegApp2OutputProcessor, self).__init__(None, None)

        self.chars_field_name = chars_field_name
        self.tag_field_name = tag_field_name

        self.new_added_field_name = new_added_field_name

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            pred_tags = ins[self.tag_field_name]
            chars = ins[self.chars_field_name]
            words = []
            start_idx = 0
            for idx, tag in enumerate(pred_tags):
                if tag==1:
                    # 当前没有考虑将原文替换回去
                    words.append(''.join(chars[start_idx:idx+1]))
                    start_idx = idx + 1
            ins[self.new_added_field_name] = ' '.join(words)


class BMES2OutputProcessor(Processor):
    """
        按照BMES标注方式推测生成的tag。由于可能存在非法tag，比如"BS"，所以需要用以下的表格做转换，cur_B意思是当前tag是B，
        next_B意思是后一个tag是B。则cur_B=S，即将当前被predict是B的tag标为S；next_M=B, 即将后一个被predict是M的tag标为B
        |       |  next_B |  next_M  |  next_E  |  next_S |   end   |
        |:-----:|:-------:|:--------:|:--------:|:-------:|:-------:|
        | start |   合法  | next_M=B | next_E=S |   合法  |    -    |
        | cur_B | cur_B=S |   合法   |   合法   | cur_B=S | cur_B=S |
        | cur_M | cur_M=E |   合法   |   合法   | cur_M=E | cur_M=E |
        | cur_E |   合法  | next_M=B | next_E=S |   合法  |   合法  |
        | cur_S |   合法  | next_M=B | next_E=S |   合法  |   合法  |
    举例：
        prediction为BSEMS，会被认为是SSSSS.

    """
    def __init__(self, chars_field_name='chars_list', tag_field_name='pred', new_added_field_name='output',
            b_idx = 0, m_idx = 1, e_idx = 2, s_idx = 3):
        """

        :param chars_field_name: character所对应的field
        :param tag_field_name: 预测对应的field
        :param new_added_field_name: 转换后的内容所在field
        :param b_idx: int, Begin标签所对应的tag idx.
        :param m_idx: int, Middle标签所对应的tag idx.
        :param e_idx: int, End标签所对应的tag idx.
        :param s_idx: int, Single标签所对应的tag idx
        """
        super(BMES2OutputProcessor, self).__init__(None, None)

        self.chars_field_name = chars_field_name
        self.tag_field_name = tag_field_name

        self.new_added_field_name = new_added_field_name

        self.b_idx = b_idx
        self.m_idx = m_idx
        self.e_idx = e_idx
        self.s_idx = s_idx
        # 还原init处介绍的矩阵
        self._valida_matrix = {
            -1: [(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1)], # magic start idx
            self.b_idx:[(0, self.s_idx), (-1, -1), (-1, -1), (0, self.s_idx), (0, self.s_idx)],
            self.m_idx:[(0, self.e_idx), (-1, -1), (-1, -1), (0, self.e_idx), (0, self.e_idx)],
            self.e_idx:[(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1), (-1, -1)],
            self.s_idx:[(-1, -1), (1, self.b_idx), (1, self.s_idx), (-1, -1), (-1, -1)],
        }

    def _validate_tags(self, tags):
        """
        给定一个tag的List，返回合法tag

        :param tags: Tensor, shape: (seq_len, )
        :return: 返回修改为合法tag的list
        """
        assert len(tags)!=0
        padded_tags = [-1, *tags, -1]
        for idx in range(len(padded_tags)-1):
            cur_tag = padded_tags[idx]
            if cur_tag not in self._valida_matrix:
                cur_tag = self.s_idx
            if padded_tags[idx+1] not in self._valida_matrix:
                padded_tags[idx+1] = self.s_idx
            next_tag = padded_tags[idx+1]
            shift_tag = self._valida_matrix[cur_tag][next_tag]
            if shift_tag[0]!=-1:
                padded_tags[idx+shift_tag[0]] = shift_tag[1]

        return padded_tags[1:-1]

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        def inner_proc(ins):
            pred_tags = ins[self.tag_field_name]
            pred_tags = self._validate_tags(pred_tags)
            chars = ins[self.chars_field_name]
            words = []
            start_idx = 0
            for idx, tag in enumerate(pred_tags):
                if tag==self.s_idx:
                    words.extend(chars[start_idx:idx+1])
                    start_idx = idx + 1
                elif tag==self.e_idx:
                    words.append(''.join(chars[start_idx:idx+1]))
                    start_idx = idx + 1
            return ' '.join(words)
        dataset.apply(func=inner_proc, new_field_name=self.new_added_field_name)


class InputTargetProcessor(Processor):
    def __init__(self, input_fields, target_fields):
        """
        对DataSet操作，将input_fields中的field设置为input，target_fields的中field设置为target

        :param input_fields: List[str], 设置为input_field的field_name。如果为None，则不将任何field设置为target。
        :param target_fields: List[str], 设置为target_field的field_name。 如果为None，则不将任何field设置为target。
        """
        super(InputTargetProcessor, self).__init__(None, None)

        if input_fields is not None and not isinstance(input_fields, list):
            raise TypeError("input_fields should be List[str], not {}.".format(type(input_fields)))
        else:
            self.input_fields = input_fields
        if target_fields is not None and not isinstance(target_fields, list):
            raise TypeError("target_fiels should be List[str], not{}.".format(type(target_fields)))
        else:
            self.target_fields = target_fields

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        if self.input_fields is not None:
            for field in self.input_fields:
                dataset.set_input(field)
        if self.target_fields is not None:
            for field in self.target_fields:
                dataset.set_target(field)

import re


from fastNLP.core.field import SeqLabelField
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.api.processor import Processor
from reproduction.chinese_word_segment.process.span_converter import SpanConverter

_SPECIAL_TAG_PATTERN = '<[a-zA-Z]+>'

class SpeicalSpanProcessor(Processor):
    # 这个类会将句子中的special span转换为对应的内容。
    def __init__(self, field_name, new_added_field_name=None):
        super(SpeicalSpanProcessor, self).__init__(field_name, new_added_field_name)

        self.span_converters = []


    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            sentence = ins[self.field_name]
            for span_converter in self.span_converters:
                sentence = span_converter.find_certain_span_and_replace(sentence)
            ins[self.new_added_field_name] = sentence

        return dataset

    def add_span_converter(self, converter):
        assert isinstance(converter, SpanConverter), "Only SpanConverterBase is allowed, not {}."\
            .format(type(converter))
        self.span_converters.append(converter)



class CWSCharSegProcessor(Processor):
    def __init__(self, field_name, new_added_field_name):
        super(CWSCharSegProcessor, self).__init__(field_name, new_added_field_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            sentence = ins[self.field_name]
            chars = self._split_sent_into_chars(sentence)
            ins[self.new_added_field_name] = chars

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
        for ins in dataset:
            sentence = ins[self.field_name]
            tag_list = self._generate_tag(sentence)
            new_tag_field = SeqLabelField(tag_list)
            ins[self.new_added_field_name] = new_tag_field
        dataset.set_is_target(**{self.new_added_field_name:True})
        return dataset

    def _tags_from_word_len(self, word_len):
        raise NotImplementedError


class CWSSegAppTagProcessor(CWSTagProcessor):
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
    def __init__(self, field_name, new_added_fielf_name=None):

        super(BigramProcessor, self).__init__(field_name, new_added_fielf_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))

        for ins in dataset:
            characters = ins[self.field_name]
            bigrams = self._generate_bigram(characters)
            ins[self.new_added_field_name] = bigrams

        return dataset


    def _generate_bigram(self, characters):
        pass


class Pre2Post2BigramProcessor(BigramProcessor):
    def __init__(self, field_name, new_added_fielf_name=None):

        super(BigramProcessor, self).__init__(field_name, new_added_fielf_name)

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


# 这里需要建立vocabulary了，但是遇到了以下的问题
# (1) 如果使用Processor的方式的话，但是在这种情况返回的不是dataset。所以建立vocabulary的工作用另外的方式实现，不借用
#   Processor了

class VocabProcessor(Processor):
    def __init__(self, field_name):

        super(VocabProcessor, self).__init__(field_name, None)
        self.vocab = Vocabulary()

    def process(self, *datasets):
        for dataset in datasets:
            assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
            for ins in dataset:
                tokens = ins[self.field_name]
                self.vocab.update(tokens)

    def get_vocab(self):
        self.vocab.build_vocab()
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)


class SeqLenProcessor(Processor):
    def __init__(self, field_name, new_added_field_name='seq_lens'):

        super(SeqLenProcessor, self).__init__(field_name, new_added_field_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            length = len(ins[self.field_name])
            ins[self.new_added_field_name] = length
        dataset.set_need_tensor(**{self.new_added_field_name:True})
        return dataset

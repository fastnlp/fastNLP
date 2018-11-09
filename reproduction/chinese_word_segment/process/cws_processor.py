
import re


from fastNLP.core.field import SeqLabelField
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet

from fastNLP.api.processor import Processor


_SPECIAL_TAG_PATTERN = '<[a-zA-Z]+>'

class FullSpaceToHalfSpaceProcessor(Processor):
    def __init__(self, field_name, change_alpha=True, change_digit=True, change_punctuation=True,
                 change_space=True):
        super(FullSpaceToHalfSpaceProcessor, self).__init__(field_name, None)

        self.change_alpha = change_alpha
        self.change_digit = change_digit
        self.change_punctuation = change_punctuation
        self.change_space = change_space

        FH_SPACE = [(u"　", u" ")]
        FH_NUM = [
            (u"０", u"0"), (u"１", u"1"), (u"２", u"2"), (u"３", u"3"), (u"４", u"4"),
            (u"５", u"5"), (u"６", u"6"), (u"７", u"7"), (u"８", u"8"), (u"９", u"9")]
        FH_ALPHA = [
            (u"ａ", u"a"), (u"ｂ", u"b"), (u"ｃ", u"c"), (u"ｄ", u"d"), (u"ｅ", u"e"),
            (u"ｆ", u"f"), (u"ｇ", u"g"), (u"ｈ", u"h"), (u"ｉ", u"i"), (u"ｊ", u"j"),
            (u"ｋ", u"k"), (u"ｌ", u"l"), (u"ｍ", u"m"), (u"ｎ", u"n"), (u"ｏ", u"o"),
            (u"ｐ", u"p"), (u"ｑ", u"q"), (u"ｒ", u"r"), (u"ｓ", u"s"), (u"ｔ", u"t"),
            (u"ｕ", u"u"), (u"ｖ", u"v"), (u"ｗ", u"w"), (u"ｘ", u"x"), (u"ｙ", u"y"),
            (u"ｚ", u"z"),
            (u"Ａ", u"A"), (u"Ｂ", u"B"), (u"Ｃ", u"C"), (u"Ｄ", u"D"), (u"Ｅ", u"E"),
            (u"Ｆ", u"F"), (u"Ｇ", u"G"), (u"Ｈ", u"H"), (u"Ｉ", u"I"), (u"Ｊ", u"J"),
            (u"Ｋ", u"K"), (u"Ｌ", u"L"), (u"Ｍ", u"M"), (u"Ｎ", u"N"), (u"Ｏ", u"O"),
            (u"Ｐ", u"P"), (u"Ｑ", u"Q"), (u"Ｒ", u"R"), (u"Ｓ", u"S"), (u"Ｔ", u"T"),
            (u"Ｕ", u"U"), (u"Ｖ", u"V"), (u"Ｗ", u"W"), (u"Ｘ", u"X"), (u"Ｙ", u"Y"),
            (u"Ｚ", u"Z")]
        # 谨慎使用标点符号转换, 因为"5．12特大地震"转换后可能就成了"5.12特大地震"
        FH_PUNCTUATION = [
            (u'％', u'%'), (u'！', u'!'), (u'＂', u'\"'), (u'＇', u'\''), (u'＃', u'#'),
            (u'￥', u'$'), (u'＆', u'&'), (u'（', u'('), (u'）', u')'), (u'＊', u'*'),
            (u'＋', u'+'), (u'，', u','), (u'－', u'-'), (u'．', u'.'), (u'／', u'/'),
            (u'：', u':'), (u'；', u';'), (u'＜', u'<'), (u'＝', u'='), (u'＞', u'>'),
            (u'？', u'?'), (u'＠', u'@'), (u'［', u'['), (u'］', u']'), (u'＼', u'\\'),
            (u'＾', u'^'), (u'＿', u'_'), (u'｀', u'`'), (u'～', u'~'), (u'｛', u'{'),
            (u'｝', u'}'), (u'｜', u'|')]
        FHs = []
        if self.change_alpha:
            FHs = FH_ALPHA
        if self.change_digit:
            FHs += FH_NUM
        if self.change_punctuation:
            FHs += FH_PUNCTUATION
        if self.change_space:
            FHs += FH_SPACE
        self.convert_map = {k: v for k, v in FHs}
    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            sentence = ins[self.field_name].text
            new_sentence = [None]*len(sentence)
            for idx, char in enumerate(sentence):
                if char in self.convert_map:
                    char = self.convert_map[char]
                new_sentence[idx] = char
            ins[self.field_name].text = ''.join(new_sentence)
        return dataset


class SpeicalSpanProcessor(Processor):
    # 这个类会将句子中的special span转换为对应的内容。
    def __init__(self, field_name, new_added_field_name=None):
        super(SpeicalSpanProcessor, self).__init__(field_name, new_added_field_name)

        self.span_converters = []


    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            sentence = ins[self.field_name].text
            for span_converter in self.span_converters:
                sentence = span_converter.find_certain_span_and_replace(sentence)
            if self.new_added_field_name!=self.field_name:
                new_text_field = TextField(sentence, is_target=False)
                ins[self.new_added_field_name] = new_text_field
            else:
                ins[self.field_name].text = sentence

        return dataset

    def add_span_converter(self, converter):
        assert isinstance(converter, SpanConverterBase), "Only SpanConverterBase is allowed, not {}."\
            .format(type(converter))
        self.span_converters.append(converter)



class CWSCharSegProcessor(Processor):
    def __init__(self, field_name, new_added_field_name):
        super(CWSCharSegProcessor, self).__init__(field_name, new_added_field_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            sentence = ins[self.field_name].text
            chars = self._split_sent_into_chars(sentence)
            new_token_field = TokenListFiled(chars, is_target=False)
            ins[self.new_added_field_name] = new_token_field

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
            sentence = ins[self.field_name].text
            tag_list = self._generate_tag(sentence)
            new_tag_field = SeqLabelField(tag_list)
            ins[self.new_added_field_name] = new_tag_field
        return dataset

    def _tags_from_word_len(self, word_len):
        raise NotImplementedError


class CWSSegAppTagProcessor(CWSTagProcessor):
    def __init__(self, field_name, new_added_field_name=None):
        super(CWSSegAppTagProcessor, self).__init__(field_name, new_added_field_name)

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
            characters = ins[self.field_name].content
            bigrams = self._generate_bigram(characters)
            new_token_field = TokenListFiled(bigrams)
            ins[self.new_added_field_name] = new_token_field

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
class IndexProcessor(Processor):
    def __init__(self, vocab, field_name):

        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        super(IndexProcessor, self).__init__(field_name, None)
        self.vocab = vocab

    def set_vocab(self, vocab):
        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        self.vocab = vocab

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            tokens = ins[self.field_name].content
            index = [self.vocab.to_index(token) for token in tokens]
            ins[self.field_name]._index = index

        return dataset


class VocabProcessor(Processor):
    def __init__(self, field_name):

        super(VocabProcessor, self).__init__(field_name, None)
        self.vocab = Vocabulary()

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            tokens = ins[self.field_name].content
            self.vocab.update(tokens)

    def get_vocab(self):
        self.vocab.build_vocab()
        return self.vocab

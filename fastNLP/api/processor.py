
from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary

class Processor:
    def __init__(self, field_name, new_added_field_name):
        self.field_name = field_name
        if new_added_field_name is None:
            self.new_added_field_name = field_name
        else:
            self.new_added_field_name = new_added_field_name

    def process(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)



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
            ins[self.field_name] = ''.join(new_sentence)
        return dataset


class IndexerProcessor(Processor):
    def __init__(self, vocab, field_name, new_added_field_name):

        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        super(IndexerProcessor, self).__init__(field_name, new_added_field_name)
        self.vocab = vocab

    def set_vocab(self, vocab):
        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        self.vocab = vocab

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            tokens = ins[self.field_name]
            index = [self.vocab.to_index(token) for token in tokens]
            ins[self.new_added_field_name] = index

        return dataset


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

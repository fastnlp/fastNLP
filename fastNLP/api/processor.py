import torch
from collections import defaultdict
import re

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler


class Processor:
    def __init__(self, field_name, new_added_field_name):
        self.field_name = field_name
        if new_added_field_name is None:
            self.new_added_field_name = field_name
        else:
            self.new_added_field_name = new_added_field_name

    def process(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)


class FullSpaceToHalfSpaceProcessor(Processor):
    """全角转半角，以字符为处理单元

    """

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
            sentence = ins[self.field_name]
            new_sentence = [None] * len(sentence)
            for idx, char in enumerate(sentence):
                if char in self.convert_map:
                    char = self.convert_map[char]
                new_sentence[idx] = char
            ins[self.field_name] = ''.join(new_sentence)
        return dataset


class PreAppendProcessor(Processor):
    def __init__(self, data, field_name, new_added_field_name=None):
        super(PreAppendProcessor, self).__init__(field_name, new_added_field_name)
        self.data = data

    def process(self, dataset):
        for ins in dataset:
            sent = ins[self.field_name]
            ins[self.new_added_field_name] = [self.data] + sent
        return dataset


class SliceProcessor(Processor):
    def __init__(self, start, end, step, field_name, new_added_field_name=None):
        super(SliceProcessor, self).__init__(field_name, new_added_field_name)
        for o in (start, end, step):
            assert isinstance(o, int) or o is None
        self.slice = slice(start, end, step)

    def process(self, dataset):
        for ins in dataset:
            sent = ins[self.field_name]
            ins[self.new_added_field_name] = sent[self.slice]
        return dataset


class Num2TagProcessor(Processor):
    def __init__(self, tag, field_name, new_added_field_name=None):
        super(Num2TagProcessor, self).__init__(field_name, new_added_field_name)
        self.tag = tag
        self.pattern = r'[-+]?([0-9]+[.]?[0-9]*)+[/eE]?[-+]?([0-9]+[.]?[0-9]*)'

    def process(self, dataset):
        for ins in dataset:
            s = ins[self.field_name]
            new_s = [None] * len(s)
            for i, w in enumerate(s):
                if re.search(self.pattern, w) is not None:
                    w = self.tag
                new_s[i] = w
            ins[self.new_added_field_name] = new_s
        return dataset


class IndexerProcessor(Processor):
    def __init__(self, vocab, field_name, new_added_field_name, delete_old_field=False):

        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        super(IndexerProcessor, self).__init__(field_name, new_added_field_name)
        self.vocab = vocab
        self.delete_old_field = delete_old_field

    def set_vocab(self, vocab):
        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        self.vocab = vocab

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            tokens = ins[self.field_name]
            index = [self.vocab.to_index(token) for token in tokens]
            ins[self.new_added_field_name] = index

        dataset._set_need_tensor(**{self.new_added_field_name: True})

        if self.delete_old_field:
            dataset.delete_field(self.field_name)

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


class SeqLenProcessor(Processor):
    def __init__(self, field_name, new_added_field_name='seq_lens'):
        super(SeqLenProcessor, self).__init__(field_name, new_added_field_name)

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        for ins in dataset:
            length = len(ins[self.field_name])
            ins[self.new_added_field_name] = length
        dataset._set_need_tensor(**{self.new_added_field_name: True})
        return dataset

class ModelProcessor(Processor):
    def __init__(self, model, seq_len_field_name='seq_lens', batch_size=32):
        """
        迭代模型并将结果的padding drop掉

        :param seq_len_field_name:
        :param batch_size:
        """
        super(ModelProcessor, self).__init__(None, None)
        self.batch_size = batch_size
        self.seq_len_field_name = seq_len_field_name
        self.model = model

    def process(self, dataset):
        self.model.eval()
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        data_iterator = Batch(dataset, batch_size=self.batch_size, sampler=SequentialSampler(), use_cuda=False)

        batch_output = defaultdict(list)
        with torch.no_grad():
            for batch_x, _ in data_iterator:
                prediction = self.model.predict(**batch_x)
                seq_lens = batch_x[self.seq_len_field_name].cpu().numpy().tolist()

                for key, value in prediction.items():
                    tmp_batch = []
                    value = value.cpu().numpy()
                    if len(value.shape) == 1 or (len(value.shape)==2 and value.shape[1]==1):
                        batch_output[key].extend(value.tolist())
                    else:
                        for idx, seq_len in enumerate(seq_lens):
                            tmp_batch.append(value[idx, :seq_len])
                        batch_output[key].extend(tmp_batch)

                batch_output[self.seq_len_field_name].extend(seq_lens)

        # TODO 当前的实现会导致之后的processor需要知道model输出的output的key是什么
        for field_name, fields in batch_output.items():
            dataset.add_field(field_name, fields, need_tensor=False, is_target=False)

        return dataset

    def set_model(self, model):
        self.model = model

    def set_model_device(self, device):
        device = torch.device(device)
        self.model.to(device)

class Index2WordProcessor(Processor):
    def __init__(self, vocab, field_name, new_added_field_name):
        super(Index2WordProcessor, self).__init__(field_name, new_added_field_name)
        self.vocab = vocab

    def process(self, dataset):
        for ins in dataset:
            new_sent = [self.vocab.to_word(w) for w in ins[self.field_name]]
            ins[self.new_added_field_name] = new_sent
        return dataset


class SetTensorProcessor(Processor):
    def __init__(self, field_dict, default=False):
        super(SetTensorProcessor, self).__init__(None, None)
        self.field_dict = field_dict
        self.default = default

    def process(self, dataset):
        set_dict = {name: self.default for name in dataset.get_fields().keys()}
        set_dict.update(self.field_dict)
        dataset._set_need_tensor(**set_dict)
        return dataset


class SetIsTargetProcessor(Processor):
    def __init__(self, field_dict, default=False):
        super(SetIsTargetProcessor, self).__init__(None, None)
        self.field_dict = field_dict
        self.default = default

    def process(self, dataset):
        set_dict = {name: self.default for name in dataset.get_fields().keys()}
        set_dict.update(self.field_dict)
        dataset.set_target(**set_dict)
        return dataset

import re
from collections import defaultdict

import torch

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.vocabulary import Vocabulary


class Processor(object):
    def __init__(self, field_name, new_added_field_name):
        """

        :param field_name: 处理哪个field
        :param new_added_field_name: 如果为None，则认为是field_name，即覆盖原有的field
        """
        self.field_name = field_name
        if new_added_field_name is None:
            self.new_added_field_name = field_name
        else:
            self.new_added_field_name = new_added_field_name

    def process(self, *args, **kwargs):
        raise NotImplementedError

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

        def inner_proc(ins):
            sentence = ins[self.field_name]
            new_sentence = [""] * len(sentence)
            for idx, char in enumerate(sentence):
                if char in self.convert_map:
                    char = self.convert_map[char]
                new_sentence[idx] = char
            return "".join(new_sentence)

        dataset.apply(inner_proc, new_field_name=self.field_name)
        return dataset


class PreAppendProcessor(Processor):
    """
    向某个field的起始增加data(应该为str类型)。该field需要为list类型。即新增的field为
        [data] + instance[field_name]

    """

    def __init__(self, data, field_name, new_added_field_name=None):
        super(PreAppendProcessor, self).__init__(field_name, new_added_field_name)
        self.data = data

    def process(self, dataset):
        dataset.apply(lambda ins: [self.data] + ins[self.field_name], new_field_name=self.new_added_field_name)
        return dataset


class SliceProcessor(Processor):
    """
    从某个field中只取部分内容。等价于instance[field_name][start:end:step]

    """

    def __init__(self, start, end, step, field_name, new_added_field_name=None):
        super(SliceProcessor, self).__init__(field_name, new_added_field_name)
        for o in (start, end, step):
            assert isinstance(o, int) or o is None
        self.slice = slice(start, end, step)

    def process(self, dataset):
        dataset.apply(lambda ins: ins[self.field_name][self.slice], new_field_name=self.new_added_field_name)
        return dataset


class Num2TagProcessor(Processor):
    """
    将一句话中的数字转换为某个tag。

    """

    def __init__(self, tag, field_name, new_added_field_name=None):
        """

        :param tag: str, 将数字转换为该tag
        :param field_name:
        :param new_added_field_name:
        """
        super(Num2TagProcessor, self).__init__(field_name, new_added_field_name)
        self.tag = tag
        self.pattern = r'[-+]?([0-9]+[.]?[0-9]*)+[/eE]?[-+]?([0-9]+[.]?[0-9]*)'

    def process(self, dataset):

        def inner_proc(ins):
            s = ins[self.field_name]
            new_s = [None] * len(s)
            for i, w in enumerate(s):
                if re.search(self.pattern, w) is not None:
                    w = self.tag
                new_s[i] = w
            return new_s

        dataset.apply(inner_proc, new_field_name=self.new_added_field_name)
        return dataset


class IndexerProcessor(Processor):
    """
    给定一个vocabulary , 将指定field转换为index形式。指定field应该是一维的list，比如
        ['我', '是', xxx]
    """

    def __init__(self, vocab, field_name, new_added_field_name, delete_old_field=False, is_input=True):

        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        super(IndexerProcessor, self).__init__(field_name, new_added_field_name)
        self.vocab = vocab
        self.delete_old_field = delete_old_field
        self.is_input = is_input

    def set_vocab(self, vocab):
        assert isinstance(vocab, Vocabulary), "Only Vocabulary class is allowed, not {}.".format(type(vocab))

        self.vocab = vocab

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only DataSet class is allowed, not {}.".format(type(dataset))
        dataset.apply(lambda ins: [self.vocab.to_index(token) for token in ins[self.field_name]],
                      new_field_name=self.new_added_field_name)
        if self.is_input:
            dataset.set_input(self.new_added_field_name)

        if self.delete_old_field:
            dataset.delete_field(self.field_name)

        return dataset


class VocabProcessor(Processor):
    """
    传入若干个DataSet以建立vocabulary。

    """

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


class SeqLenProcessor(Processor):
    """
    根据某个field新增一个sequence length的field。取该field的第一维

    """

    def __init__(self, field_name, new_added_field_name='seq_lens', is_input=True):
        super(SeqLenProcessor, self).__init__(field_name, new_added_field_name)
        self.is_input = is_input

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
        dataset.apply(lambda ins: len(ins[self.field_name]), new_field_name=self.new_added_field_name)
        if self.is_input:
            dataset.set_input(self.new_added_field_name)
        return dataset


from fastNLP.core.utils import _build_args


class ModelProcessor(Processor):
    def __init__(self, model, seq_len_field_name='seq_lens', batch_size=32):
        """
        传入一个model，在process()时传入一个dataset，该processor会通过Batch将DataSet的内容输出给model.predict或者model.forward.
            model输出的内容会被增加到dataset中，field_name由model输出决定。如果生成的内容维度不是(Batch_size, )与
            (Batch_size, 1)，则使用seqence  length这个field进行unpad
        TODO 这个类需要删除对seq_lens的依赖。

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
        data_iterator = Batch(dataset, batch_size=self.batch_size, sampler=SequentialSampler())

        batch_output = defaultdict(list)
        if hasattr(self.model, "predict"):
            predict_func = self.model.predict
        else:
            predict_func = self.model.forward
        with torch.no_grad():
            for batch_x, _ in data_iterator:
                refined_batch_x = _build_args(predict_func, **batch_x)
                prediction = predict_func(**refined_batch_x)
                seq_lens = batch_x[self.seq_len_field_name].tolist()

                for key, value in prediction.items():
                    tmp_batch = []
                    value = value.cpu().numpy()
                    if len(value.shape) == 1 or (len(value.shape) == 2 and value.shape[1] == 1):
                        batch_output[key].extend(value.tolist())
                    else:
                        for idx, seq_len in enumerate(seq_lens):
                            tmp_batch.append(value[idx, :seq_len])
                        batch_output[key].extend(tmp_batch)
                if not self.seq_len_field_name in prediction:
                    batch_output[self.seq_len_field_name].extend(seq_lens)

        # TODO 当前的实现会导致之后的processor需要知道model输出的output的key是什么
        for field_name, fields in batch_output.items():
            dataset.add_field(field_name, fields, is_input=True, is_target=False)

        return dataset

    def set_model(self, model):
        self.model = model

    def set_model_device(self, device):
        device = torch.device(device)
        self.model.to(device)


class Index2WordProcessor(Processor):
    """
    将DataSet中某个为index的field根据vocab转换为str

    """

    def __init__(self, vocab, field_name, new_added_field_name):
        super(Index2WordProcessor, self).__init__(field_name, new_added_field_name)
        self.vocab = vocab

    def process(self, dataset):
        dataset.apply(lambda ins: [self.vocab.to_word(w) for w in ins[self.field_name]],
                      new_field_name=self.new_added_field_name)
        return dataset


class SetTargetProcessor(Processor):
    def __init__(self, *fields, flag=True):
        super(SetTargetProcessor, self).__init__(None, None)
        self.fields = fields
        self.flag = flag

    def process(self, dataset):
        dataset.set_target(*self.fields, flag=self.flag)
        return dataset


class SetInputProcessor(Processor):
    def __init__(self, *fields, flag=True):
        super(SetInputProcessor, self).__init__(None, None)
        self.fields = fields
        self.flag = flag

    def process(self, dataset):
        dataset.set_input(*self.fields, flag=self.flag)
        return dataset


class VocabIndexerProcessor(Processor):
    """
    根据DataSet创建Vocabulary，并将其用数字index。新生成的index的field会被放在new_added_filed_name, 如果没有提供
        new_added_field_name, 则覆盖原有的field_name.

    """

    def __init__(self, field_name, new_added_filed_name=None, min_freq=1, max_size=None,
                 verbose=0, is_input=True):
        """

        :param field_name: 从哪个field_name创建词表，以及对哪个field_name进行index操作
        :param new_added_filed_name: index时，生成的index field的名称，如果不传入，则覆盖field_name.
        :param min_freq: 创建的Vocabulary允许的单词最少出现次数.
        :param max_size: 创建的Vocabulary允许的最大的单词数量
        :param verbose: 0, 不输出任何信息；1，输出信息
        :param bool is_input:
        """
        super(VocabIndexerProcessor, self).__init__(field_name, new_added_filed_name)
        self.min_freq = min_freq
        self.max_size = max_size

        self.verbose = verbose
        self.is_input = is_input

    def construct_vocab(self, *datasets):
        """
        使用传入的DataSet创建vocabulary

        :param datasets: DataSet类型的数据，用于构建vocabulary
        :return:
        """
        self.vocab = Vocabulary(min_freq=self.min_freq, max_size=self.max_size)
        for dataset in datasets:
            assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))
            dataset.apply(lambda ins: self.vocab.update(ins[self.field_name]))
        self.vocab.build_vocab()
        if self.verbose:
            print("Vocabulary Constructed, has {} items.".format(len(self.vocab)))

    def process(self, *datasets, only_index_dataset=None):
        """
        若还未建立Vocabulary，则使用dataset中的DataSet建立vocabulary；若已经有了vocabulary则使用已有的vocabulary。得到vocabulary
            后，则会index datasets与only_index_dataset。

        :param datasets: DataSet类型的数据
        :param only_index_dataset: DataSet, or list of DataSet. 该参数中的内容只会被用于index，不会被用于生成vocabulary。
        :return:
        """
        if len(datasets) == 0 and not hasattr(self, 'vocab'):
            raise RuntimeError("You have to construct vocabulary first. Or you have to pass datasets to construct it.")
        if not hasattr(self, 'vocab'):
            self.construct_vocab(*datasets)
        else:
            if self.verbose:
                print("Using constructed vocabulary with {} items.".format(len(self.vocab)))
        to_index_datasets = []
        if len(datasets) != 0:
            for dataset in datasets:
                assert isinstance(dataset, DataSet), "Only DataSet class is allowed, not {}.".format(type(dataset))
                to_index_datasets.append(dataset)

        if not (only_index_dataset is None):
            if isinstance(only_index_dataset, list):
                for dataset in only_index_dataset:
                    assert isinstance(dataset, DataSet), "Only DataSet class is allowed, not {}.".format(type(dataset))
                    to_index_datasets.append(dataset)
            elif isinstance(only_index_dataset, DataSet):
                to_index_datasets.append(only_index_dataset)
            else:
                raise TypeError('Only DataSet or list of DataSet is allowed, not {}.'.format(type(only_index_dataset)))

        for dataset in to_index_datasets:
            assert isinstance(dataset, DataSet), "Only DataSet class is allowed, not {}.".format(type(dataset))
            dataset.apply(lambda ins: [self.vocab.to_index(token) for token in ins[self.field_name]],
                          new_field_name=self.new_added_field_name, is_input=self.is_input)
        # 只返回一个，infer时为了跟其他processor保持一致
        if len(to_index_datasets) == 1:
            return to_index_datasets[0]

    def set_vocab(self, vocab):
        assert isinstance(vocab, Vocabulary), "Only fastNLP.core.Vocabulary is allowed, not {}.".format(type(vocab))
        self.vocab = vocab

    def delete_vocab(self):
        del self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def set_verbose(self, verbose):
        """
        设置processor verbose状态。

        :param verbose: int, 0，不输出任何信息；1，输出vocab 信息。
        :return:
        """
        self.verbose = verbose

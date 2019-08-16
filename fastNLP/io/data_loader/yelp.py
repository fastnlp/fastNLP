
import csv
from typing import Iterable

from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance
from ...core.vocabulary import VocabularyOption, Vocabulary
from ..base_loader import DataBundle, DataSetLoader
from typing import Union, Dict
from ..utils import check_loader_paths, get_tokenizer


class YelpLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.YelpLoader` :class:`fastNLP.io.data_loader.YelpLoader`
    读取Yelp_full/Yelp_polarity数据集, DataSet包含fields:

        words: list(str), 需要分类的文本

        target: str, 文本的标签

        chars:list(str),未index的字符列表

    数据集：yelp_full/yelp_polarity

    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    :param lower: 是否需要自动转小写，默认为False。
    """

    def __init__(self, fine_grained=False, lower=False):
        super(YelpLoader, self).__init__()
        tag_v = {'1.0': 'very negative', '2.0': 'negative', '3.0': 'neutral',
                 '4.0': 'positive', '5.0': 'very positive'}
        if not fine_grained:
            tag_v['1.0'] = tag_v['2.0']
            tag_v['5.0'] = tag_v['4.0']
        self.fine_grained = fine_grained
        self.tag_v = tag_v
        self.lower = lower
        self.tokenizer = get_tokenizer()

    def _load(self, path):
        ds = DataSet()
        csv_reader = csv.reader(open(path, encoding='utf-8'))
        all_count = 0
        real_count = 0
        for row in csv_reader:
            all_count += 1
            if len(row) == 2:
                target = self.tag_v[row[0] + ".0"]
                words = clean_str(row[1], self.tokenizer, self.lower)
                if len(words) != 0:
                    ds.append(Instance(words=words, target=target))
                    real_count += 1
        print("all count:", all_count)
        print("real count:", real_count)
        return ds

    def process(self, paths: Union[str, Dict[str, str]],
                train_ds: Iterable[str] = None,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,
                char_level_op=False):
        paths = check_loader_paths(paths)
        info = DataBundle(datasets=self.load(paths))
        src_vocab = Vocabulary() if src_vocab_op is None else Vocabulary(**src_vocab_op)
        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_op is None else Vocabulary(**tgt_vocab_op)
        _train_ds = [info.datasets[name]
                     for name in train_ds] if train_ds else info.datasets.values()

        def wordtochar(words):
            chars = []
            for word in words:
                word = word.lower()
                for char in word:
                    chars.append(char)
                chars.append('')
            chars.pop()
            return chars

        input_name, target_name = Const.INPUT, Const.TARGET
        info.vocabs = {}
        # 就分隔为char形式
        if char_level_op:
            for dataset in info.datasets.values():
                dataset.apply_field(wordtochar, field_name=Const.INPUT, new_field_name=Const.CHAR_INPUT)
        else:
            src_vocab.from_dataset(*_train_ds, field_name=input_name)
            src_vocab.index_dataset(*info.datasets.values(), field_name=input_name, new_field_name=input_name)
            info.vocabs[input_name] = src_vocab

        tgt_vocab.from_dataset(*_train_ds, field_name=target_name)
        tgt_vocab.index_dataset(
            *info.datasets.values(),
            field_name=target_name, new_field_name=target_name)

        info.vocabs[target_name] = tgt_vocab

        info.datasets['train'], info.datasets['dev'] = info.datasets['train'].split(0.1, shuffle=False)

        for name, dataset in info.datasets.items():
            dataset.set_input(Const.INPUT)
            dataset.set_target(Const.TARGET)

        return info


def clean_str(sentence, tokenizer, char_lower=False):
    """
    heavily borrowed from github
    https://github.com/LukeZhuang/Hierarchical-Attention-Network/blob/master/yelp-preprocess.ipynb
    :param sentence:  is a str
    :return:
    """
    if char_lower:
        sentence = sentence.lower()
    import re
    nonalpnum = re.compile('[^0-9a-zA-Z?!\']+')
    words = tokenizer(sentence)
    words_collection = []
    for word in words:
        if word in ['-lrb-', '-rrb-', '<sssss>', '-r', '-l', 'b-']:
            continue
        tt = nonalpnum.split(word)
        t = ''.join(tt)
        if t != '':
            words_collection.append(t)

    return words_collection


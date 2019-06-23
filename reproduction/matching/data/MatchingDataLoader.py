
import os

from nltk import Tree
from typing import Union, Dict

from fastNLP.core.const import Const
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.io.base_loader import DataInfo
from fastNLP.io.dataset_loader import JsonLoader
from fastNLP.io.file_utils import _get_base_url, cached_path
from fastNLP.modules.encoder._bert import BertTokenizer


class MatchingLoader(JsonLoader):
    """
    别名：:class:`fastNLP.io.MatchingLoader` :class:`fastNLP.io.dataset_loader.MatchingLoader`

    读取Matching任务的数据集
    """

    def __init__(self, fields=None, paths: dict=None):
        super(MatchingLoader, self).__init__(fields=fields)
        self.paths = paths

    def _load(self, path):
        return super(MatchingLoader, self)._load(path)

    def process(self, paths: Union[str, Dict[str, str]], dataset_name=None,
                to_lower=False, char_information=False, seq_len_type: str=None,
                bert_tokenizer: str=None, get_index=True, set_input: Union[list, str, bool]=True,
                set_target: Union[list, str, bool] = True, concat: Union[str, list, bool]=None, ) -> DataInfo:
        if isinstance(set_input, str):
            set_input = [set_input]
        if isinstance(set_target, str):
            set_target = [set_target]
        if isinstance(set_input, bool):
            auto_set_input = set_input
        else:
            auto_set_input = False
        if isinstance(set_target, bool):
            auto_set_target = set_target
        else:
            auto_set_target = False
        if isinstance(paths, str):
            if os.path.isdir(paths):
                path = {n: os.path.join(paths, self.paths[n]) for n in self.paths.keys()}
            else:
                path = {dataset_name if dataset_name is not None else 'train': paths}
        else:
            path = paths

        data_info = DataInfo()
        for data_name in path.keys():
            data_info.datasets[data_name] = self._load(path[data_name])

        for data_name, data_set in data_info.datasets.items():
            if auto_set_input:
                data_set.set_input(Const.INPUTS(0), Const.INPUTS(1))
            if auto_set_target:
                data_set.set_target(Const.TARGET)

        if to_lower:
            for data_name, data_set in data_info.datasets.items():
                data_set.apply(lambda x: [w.lower() for w in x[Const.INPUTS(0)]], new_field_name=Const.INPUTS(0),
                               is_input=auto_set_input)
                data_set.apply(lambda x: [w.lower() for w in x[Const.INPUTS(1)]], new_field_name=Const.INPUTS(1),
                               is_input=auto_set_input)

        if bert_tokenizer is not None:
            PRETRAINED_BERT_MODEL_DIR = {'en': 'bert-base-cased-f89bfe08.zip',
                                         'en-base-uncased': 'bert-base-uncased-3413b23c.zip',
                                         'en-base-cased': 'bert-base-cased-f89bfe08.zip',
                                         'en-large-uncased': 'bert-large-uncased-20939f45.zip',
                                         'en-large-cased': 'bert-large-cased-e0cf90fc.zip',

                                         'cn': 'bert-base-chinese-29d0a84a.zip',
                                         'cn-base': 'bert-base-chinese-29d0a84a.zip',

                                         'multilingual': 'bert-base-multilingual-cased-1bd364ee.zip',
                                         'multilingual-base-uncased': 'bert-base-multilingual-uncased-f8730fe4.zip',
                                         'multilingual-base-cased': 'bert-base-multilingual-cased-1bd364ee.zip',
                                         }
            if bert_tokenizer.lower() in PRETRAINED_BERT_MODEL_DIR:
                PRETRAIN_URL = _get_base_url('bert')
                model_name = PRETRAINED_BERT_MODEL_DIR[bert_tokenizer]
                model_url = PRETRAIN_URL + model_name
                model_dir = cached_path(model_url)
                # 检查是否存在
            elif os.path.isdir(bert_tokenizer):
                model_dir = bert_tokenizer
            else:
                raise ValueError(f"Cannot recognize BERT tokenizer from {bert_tokenizer}.")

            tokenizer = BertTokenizer.from_pretrained(model_dir)

            for data_name, data_set in data_info.datasets.items():
                for fields in data_set.get_field_names():
                    if Const.INPUT in fields:
                        data_set.apply(lambda x: tokenizer.tokenize(' '.join(x[fields])), new_field_name=fields,
                                       is_input=auto_set_input)

        if isinstance(concat, bool):
            concat = 'default' if concat else None
        if concat is not None:
            if isinstance(concat, str):
                CONCAT_MAP = {'bert': ['[CLS]', '[SEP]', '', '[SEP]'],
                              'default': ['', '<sep>', '', '']}
                if concat.lower() in CONCAT_MAP:
                    concat = CONCAT_MAP[concat]
                else:
                    concat = 4 * [concat]
            assert len(concat) == 4, \
                f'Please choose a list with 4 symbols which at the beginning of first sentence ' \
                f'the end of first sentence, the begin of second sentence, and the end of second' \
                f'sentence. Your input is {concat}'

            for data_name, data_set in data_info.datasets.items():
                data_set.apply(lambda x: [concat[0]] + x[Const.INPUTS(0)] + [concat[1]] + [concat[2]] +
                               x[Const.INPUTS(1)] + [concat[3]], new_field_name=Const.INPUT)
                data_set.apply(lambda x: [w for w in x[Const.INPUT] if len(w) > 0], new_field_name=Const.INPUT,
                               is_input=auto_set_input)

        if seq_len_type is not None:
            if seq_len_type == 'seq_len':  #
                for data_name, data_set in data_info.datasets.items():
                    for fields in data_set.get_field_names():
                        if Const.INPUT in fields:
                            data_set.apply(lambda x: len(x[fields]),
                                           new_field_name=fields.replace(Const.INPUT, Const.TARGET),
                                           is_input=auto_set_input)
            elif seq_len_type == 'mask':
                for data_name, data_set in data_info.datasets.items():
                    for fields in data_set.get_field_names():
                        if Const.INPUT in fields:
                            data_set.apply(lambda x: [1] * len(x[fields]),
                                           new_field_name=fields.replace(Const.INPUT, Const.TARGET),
                                           is_input=auto_set_input)
            elif seq_len_type == 'bert':
                for data_name, data_set in data_info.datasets.items():
                    if Const.INPUT not in data_set.get_field_names():
                        raise KeyError(f'Field ``{Const.INPUT}`` not in {data_name} data set: '
                                       f'got {data_set.get_field_names()}')
                    data_set.apply(lambda x: [0] * (len(x[Const.INPUTS(0)]) + 2) + [1] * (len(x[Const.INPUTS(1)]) + 1),
                                   new_field_name=Const.INPUT_LENS(0), is_input=auto_set_input)
                    data_set.apply(lambda x: [1] * len(x[Const.INPUT_LENS(0)]),
                                   new_field_name=Const.INPUT_LENS(1), is_input=auto_set_input)

        data_set_list = [d for n, d in data_info.datasets.items()]
        assert len(data_set_list) > 0, f'There are NO data sets in data info!'

        if bert_tokenizer is not None:
            words_vocab = Vocabulary(padding='[PAD]', unknown='[UNK]')
        else:
            words_vocab = Vocabulary()
        words_vocab = words_vocab.from_dataset(*data_set_list,
                                               field_name=[n for n in data_set_list[0].get_field_names()
                                                           if (Const.INPUT in n)])
        target_vocab = Vocabulary(padding=None, unknown=None)
        target_vocab = target_vocab.from_dataset(*data_set_list, field_name=Const.TARGET)
        data_info.vocabs = {Const.INPUT: words_vocab, Const.TARGET: target_vocab}

        if get_index:
            for data_name, data_set in data_info.datasets.items():
                for fields in data_set.get_field_names():
                    if Const.INPUT in fields:
                        data_set.apply(lambda x: [words_vocab.to_index(w) for w in x[fields]], new_field_name=fields,
                                       is_input=auto_set_input)

                data_set.apply(lambda x: target_vocab.to_index(x[Const.TARGET]), new_field_name=Const.TARGET,
                               is_input=auto_set_input, is_target=auto_set_target)

        for data_name, data_set in data_info.datasets.items():
            if isinstance(set_input, list):
                data_set.set_input(set_input)
            if isinstance(set_target, list):
                data_set.set_target(set_target)

        return data_info


class SNLILoader(MatchingLoader):
    """
    别名：:class:`fastNLP.io.SNLILoader` :class:`fastNLP.io.dataset_loader.SNLILoader`

    读取SNLI数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise
        words2: list(str), 第二句文本, hypothesis
        target: str, 真实标签

    数据来源: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    """

    def __init__(self, paths: dict=None):
        fields = {
            'sentence1_parse': Const.INPUTS(0),
            'sentence2_parse': Const.INPUTS(1),
            'gold_label': Const.TARGET,
        }
        paths = paths if paths is not None else {
            'train': 'snli_1.0_train.jsonl',
            'dev': 'snli_1.0_dev.jsonl',
            'test': 'snli_1.0_test.jsonl'}
        super(SNLILoader, self).__init__(fields=fields, paths=paths)

    def _load(self, path):
        ds = super(SNLILoader, self)._load(path)

        def parse_tree(x):
            t = Tree.fromstring(x)
            return t.leaves()

        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(0)]), new_field_name=Const.INPUTS(0))
        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(1)]), new_field_name=Const.INPUTS(1))
        ds.drop(lambda x: x[Const.TARGET] == '-')
        return ds




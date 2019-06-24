
import os

from nltk import Tree
from typing import Union, Dict

from fastNLP.core.const import Const
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.base_loader import DataInfo
from fastNLP.io.dataset_loader import JsonLoader, DataSetLoader
from fastNLP.io.file_utils import _get_base_url, cached_path, PRETRAINED_BERT_MODEL_DIR
from fastNLP.modules.encoder._bert import BertTokenizer


class MatchingLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.MatchingLoader` :class:`fastNLP.io.dataset_loader.MatchingLoader`

    读取Matching任务的数据集
    """

    def __init__(self, paths: dict=None):
        """
        :param dict paths: key是数据集名称（如train、dev、test），value是对应的文件名
        """
        self.paths = paths

    def _load(self, path):
        """
        :param str path: 待读取数据集的路径名
        :return: fastNLP.DataSet ds: 返回一个DataSet对象，里面必须包含3个field：其中两个分别为两个句子
            的原始字符串文本，第三个为标签
        """
        raise NotImplementedError

    def process(self, paths: Union[str, Dict[str, str]], dataset_name: str=None,
                to_lower=False, seq_len_type: str=None, bert_tokenizer: str=None,
                get_index=True, set_input: Union[list, str, bool]=True,
                set_target: Union[list, str, bool] = True, concat: Union[str, list, bool]=None, ) -> DataInfo:
        """
        :param paths: str或者Dict[str, str]。如果是str，则为数据集所在的文件夹或者是全路径文件名：如果是文件夹，
            则会从self.paths里面找对应的数据集名称与文件名。如果是Dict，则为数据集名称（如train、dev、test）和
            对应的全路径文件名。
        :param str dataset_name: 如果在paths里传入的是一个数据集的全路径文件名，那么可以用dataset_name来定义
            这个数据集的名字，如果不定义则默认为train。
        :param bool to_lower: 是否将文本自动转为小写。默认值为False。
        :param str seq_len_type: 提供的seq_len类型，支持 ``seq_len`` ：提供一个数字作为句子长度； ``mask`` :
            提供一个0/1的mask矩阵作为句子长度； ``bert`` ：提供segment_type_id（第一个句子为0，第二个句子为1）和
            attention mask矩阵（0/1的mask矩阵）。默认值为None，即不提供seq_len
        :param str bert_tokenizer: bert tokenizer所使用的词表所在的文件夹路径
        :param bool get_index: 是否需要根据词表将文本转为index
        :param set_input: 如果为True，则会自动将相关的field（名字里含有Const.INPUT的）设置为input，如果为False
            则不会将任何field设置为input。如果传入str或者List[str]，则会根据传入的内容将相对应的field设置为input，
            于此同时其他field不会被设置为input。默认值为True。
        :param set_target: set_target将控制哪些field可以被设置为target，用法与set_input一致。默认值为True。
        :param concat: 是否需要将两个句子拼接起来。如果为False则不会拼接。如果为True则会在两个句子之间插入一个<sep>。
            如果传入一个长度为4的list，则分别表示插在第一句开始前、第一句结束后、第二句开始前、第二句结束后的标识符。如果
            传入字符串 ``bert`` ，则会采用bert的拼接方式，等价于['[CLS]', '[SEP]', '', '[SEP]'].
        :return:
        """
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
                                           new_field_name=fields.replace(Const.INPUT, Const.INPUT_LEN),
                                           is_input=auto_set_input)
            elif seq_len_type == 'mask':
                for data_name, data_set in data_info.datasets.items():
                    for fields in data_set.get_field_names():
                        if Const.INPUT in fields:
                            data_set.apply(lambda x: [1] * len(x[fields]),
                                           new_field_name=fields.replace(Const.INPUT, Const.INPUT_LEN),
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
            with open(os.path.join(model_dir, 'vocab.txt'), 'r') as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            words_vocab.add_word_lst(lines)
            words_vocab.build_vocab()
        else:
            words_vocab = Vocabulary()
            words_vocab = words_vocab.from_dataset(*[d for n, d in data_info.datasets.items() if 'train' in n],
                                                   field_name=[n for n in data_set_list[0].get_field_names()
                                                               if (Const.INPUT in n)],
                                                   no_create_entry_dataset=[d for n, d in data_info.datasets.items()
                                                                            if 'train' not in n])
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
                data_set.set_input(*set_input)
            if isinstance(set_target, list):
                data_set.set_target(*set_target)

        return data_info


class SNLILoader(MatchingLoader, JsonLoader):
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
        # super(SNLILoader, self).__init__(fields=fields, paths=paths)
        MatchingLoader.__init__(self, paths=paths)
        JsonLoader.__init__(self, fields=fields)

    def _load(self, path):
        # ds = super(SNLILoader, self)._load(path)
        ds = JsonLoader._load(self, path)

        def parse_tree(x):
            t = Tree.fromstring(x)
            return t.leaves()

        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(0)]), new_field_name=Const.INPUTS(0))
        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(1)]), new_field_name=Const.INPUTS(1))
        ds.drop(lambda x: x[Const.TARGET] == '-')
        return ds




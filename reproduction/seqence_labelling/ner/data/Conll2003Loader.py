
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io.base_loader import DataSetLoader, DataInfo
from typing import Union, Dict
from fastNLP import Vocabulary
from fastNLP import Const
from reproduction.utils import check_dataloader_paths

from fastNLP.io import ConllLoader
from reproduction.seqence_labelling.ner.data.utils import iob2bioes, iob2


class Conll2003DataLoader(DataSetLoader):
    def __init__(self, task:str='ner', encoding_type:str='bioes'):
        """
        加载Conll2003格式的英语语料，该数据集的信息可以在https://www.clips.uantwerpen.be/conll2003/ner/找到。当task为pos
            时，返回的DataSet中target取值于第2列; 当task为chunk时，返回的DataSet中target取值于第3列;当task为ner时，返回
            的DataSet中target取值于第4列。所有"-DOCSTART- -X- O O"将被忽略，这会导致数据的数量少于很多文献报道的值，但
            鉴于"-DOCSTART- -X- O O"只是用于文档分割的符号，并不应该作为预测对象，所以我们忽略了数据中的-DOCTSTART-开头的行
        ner与chunk任务读取后的数据的target将为encoding_type类型。pos任务读取后就是pos列的数据。

        :param task: 指定需要标注任务。可选ner, pos, chunk
        """
        assert task in ('ner', 'pos', 'chunk')
        index = {'ner':3, 'pos':1, 'chunk':2}[task]
        self._loader = ConllLoader(headers=['raw_words', 'target'], indexes=[0, index])
        self._tag_converters = []
        if task in ('ner', 'chunk'):
            self._tag_converters = [iob2]
            if encoding_type == 'bioes':
                self._tag_converters.append(iob2bioes)

    def load(self, path: str):
        dataset = self._loader.load(path)
        def convert_tag_schema(tags):
            for converter in self._tag_converters:
                tags = converter(tags)
            return tags
        if self._tag_converters:
            dataset.apply_field(convert_tag_schema, field_name=Const.TARGET, new_field_name=Const.TARGET)
        return dataset

    def process(self, paths: Union[str, Dict[str, str]], word_vocab_opt:VocabularyOption=None, lower:bool=False):
        """
        读取并处理数据。数据中的'-DOCSTART-'开头的行会被忽略

        :param paths:
        :param word_vocab_opt: vocabulary的初始化值
        :param lower: 是否将所有字母转为小写。
        :return:
        """
        # 读取数据
        paths = check_dataloader_paths(paths)
        data = DataInfo()
        input_fields = [Const.TARGET, Const.INPUT, Const.INPUT_LEN]
        target_fields = [Const.TARGET, Const.INPUT_LEN]
        for name, path in paths.items():
            dataset = self.load(path)
            dataset.apply_field(lambda words: words, field_name='raw_words', new_field_name=Const.INPUT)
            if lower:
                dataset.words.lower()
            data.datasets[name] = dataset

        # 对construct vocab
        word_vocab = Vocabulary(min_freq=2) if word_vocab_opt is None else Vocabulary(**word_vocab_opt)
        word_vocab.from_dataset(data.datasets['train'], field_name=Const.INPUT,
                                no_create_entry_dataset=[dataset for name, dataset in data.datasets.items() if name!='train'])
        word_vocab.index_dataset(*data.datasets.values(), field_name=Const.INPUT, new_field_name=Const.INPUT)
        data.vocabs[Const.INPUT] = word_vocab

        # cap words
        cap_word_vocab = Vocabulary()
        cap_word_vocab.from_dataset(data.datasets['train'], field_name='raw_words',
                                no_create_entry_dataset=[dataset for name, dataset in data.datasets.items() if name!='train'])
        cap_word_vocab.index_dataset(*data.datasets.values(), field_name='raw_words', new_field_name='cap_words')
        input_fields.append('cap_words')
        data.vocabs['cap_words'] = cap_word_vocab

        # 对target建vocab
        target_vocab = Vocabulary(unknown=None, padding=None)
        target_vocab.from_dataset(*data.datasets.values(), field_name=Const.TARGET)
        target_vocab.index_dataset(*data.datasets.values(), field_name=Const.TARGET)
        data.vocabs[Const.TARGET] = target_vocab

        for name, dataset in data.datasets.items():
            dataset.add_seq_len(Const.INPUT, new_field_name=Const.INPUT_LEN)
            dataset.set_input(*input_fields)
            dataset.set_target(*target_fields)

        return data

if __name__ == '__main__':
    pass
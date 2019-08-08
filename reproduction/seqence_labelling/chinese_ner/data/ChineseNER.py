

from fastNLP.io.base_loader import DataSetLoader, DataBundle
from fastNLP.io import ConllLoader
from reproduction.seqence_labelling.ner.data.utils import iob2bioes, iob2
from fastNLP import Const
from reproduction.utils import check_dataloader_paths
from fastNLP import Vocabulary

class ChineseNERLoader(DataSetLoader):
    """
    读取中文命名实体数据集，包括PeopleDaily, MSRA-NER, Weibo。数据在这里可以找到https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER
    请确保输入数据的格式如下, 共两列，第一列为字，第二列为标签，不同句子以空行隔开
        我 O
        们 O
        变 O
        而 O
        以 O
        书 O
        会 O
        ...

    """
    def __init__(self, encoding_type:str='bioes'):
        """

        :param str encoding_type: 支持bio和bioes格式
        """
        super().__init__()
        self._loader = ConllLoader(headers=['raw_chars', 'target'], indexes=[0, 1])

        assert encoding_type in ('bio', 'bioes')

        self._tag_converters = [iob2]
        if encoding_type == 'bioes':
            self._tag_converters.append(iob2bioes)

    def load(self, path:str):
        dataset = self._loader.load(path)
        def convert_tag_schema(tags):
            for converter in self._tag_converters:
                tags = converter(tags)
            return tags
        if self._tag_converters:
            dataset.apply_field(convert_tag_schema, field_name=Const.TARGET, new_field_name=Const.TARGET)
        return dataset

    def process(self, paths, bigrams=False, trigrams=False):
        """

        :param paths:
        :param bool, bigrams: 是否包含生成bigram feature, [a, b, c, d] -> [ab, bc, cd, d<eos>]
        :param bool, trigrams: 是否包含trigram feature，[a, b, c, d] -> [abc, bcd, cd<eos>, d<eos><eos>]
        :return: DataBundle
            包含以下的fields
                raw_chars: List[str]
                chars: List[int]
                seq_len: int, 字的长度
                bigrams: List[int], optional
                trigrams: List[int], optional
                target: List[int]
        """
        paths = check_dataloader_paths(paths)
        data = DataBundle()
        input_fields = [Const.CHAR_INPUT, Const.INPUT_LEN, Const.TARGET]
        target_fields = [Const.TARGET, Const.INPUT_LEN]

        for name, path in paths.items():
            dataset = self.load(path)
            if bigrams:
                dataset.apply_field(lambda raw_chars: [c1+c2 for c1, c2 in zip(raw_chars, raw_chars[1:]+['<eos>'])],
                                    field_name='raw_chars', new_field_name='bigrams')

            if trigrams:
                dataset.apply_field(lambda raw_chars: [c1+c2+c3 for c1, c2, c3 in zip(raw_chars,
                                                                                      raw_chars[1:]+['<eos>'],
                                                                                      raw_chars[2:]+['<eos>']*2)],
                                    field_name='raw_chars', new_field_name='trigrams')
            data.datasets[name] = dataset

        char_vocab = Vocabulary().from_dataset(data.datasets['train'], field_name='raw_chars',
                                no_create_entry_dataset=[dataset for name, dataset in data.datasets.items() if name!='train'])
        char_vocab.index_dataset(*data.datasets.values(), field_name='raw_chars', new_field_name=Const.CHAR_INPUT)
        data.vocabs[Const.CHAR_INPUT] = char_vocab

        target_vocab = Vocabulary(unknown=None, padding=None).from_dataset(data.datasets['train'], field_name=Const.TARGET)
        target_vocab.index_dataset(*data.datasets.values(), field_name=Const.TARGET)
        data.vocabs[Const.TARGET] = target_vocab

        if bigrams:
            bigram_vocab = Vocabulary().from_dataset(data.datasets['train'], field_name='bigrams',
                                                   no_create_entry_dataset=[dataset for name, dataset in
                                                                            data.datasets.items() if name != 'train'])
            bigram_vocab.index_dataset(*data.datasets.values(), field_name='bigrams', new_field_name='bigrams')
            data.vocabs['bigrams'] = bigram_vocab
            input_fields.append('bigrams')

        if trigrams:
            trigram_vocab = Vocabulary().from_dataset(data.datasets['train'], field_name='trigrams',
                                                      no_create_entry_dataset=[dataset for name, dataset in
                                                                               data.datasets.items() if name != 'train'])
            trigram_vocab.index_dataset(*data.datasets.values(), field_name='trigrams', new_field_name='trigrams')
            data.vocabs['trigrams'] = trigram_vocab
            input_fields.append('trigrams')

        for name, dataset in data.datasets.items():
            dataset.add_seq_len(Const.CHAR_INPUT)
            dataset.set_input(*input_fields)
            dataset.set_target(*target_fields)

        return data





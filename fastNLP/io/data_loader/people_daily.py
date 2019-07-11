
from ..base_loader import DataSetLoader
from ...core.dataset import DataSet
from ...core.instance import Instance
from ...core.const import Const


class PeopleDailyCorpusLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.PeopleDailyCorpusLoader` :class:`fastNLP.io.dataset_loader.PeopleDailyCorpusLoader`

    读取人民日报数据集
    """

    def __init__(self, pos=True, ner=True):
        super(PeopleDailyCorpusLoader, self).__init__()
        self.pos = pos
        self.ner = ner

    def _load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            sents = f.readlines()
        examples = []
        for sent in sents:
            if len(sent) <= 2:
                continue
            inside_ne = False
            sent_pos_tag = []
            sent_words = []
            sent_ner = []
            words = sent.strip().split()[1:]
            for word in words:
                if "[" in word and "]" in word:
                    ner_tag = "U"
                    print(word)
                elif "[" in word:
                    inside_ne = True
                    ner_tag = "B"
                    word = word[1:]
                elif "]" in word:
                    ner_tag = "L"
                    word = word[:word.index("]")]
                    if inside_ne is True:
                        inside_ne = False
                    else:
                        raise RuntimeError("only ] appears!")
                else:
                    if inside_ne is True:
                        ner_tag = "I"
                    else:
                        ner_tag = "O"
                tmp = word.split("/")
                token, pos = tmp[0], tmp[1]
                sent_ner.append(ner_tag)
                sent_pos_tag.append(pos)
                sent_words.append(token)
            example = [sent_words]
            if self.pos is True:
                example.append(sent_pos_tag)
            if self.ner is True:
                example.append(sent_ner)
            examples.append(example)
        return self.convert(examples)

    def convert(self, data):
        """

        :param data: python 内置对象
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        data_set = DataSet()
        for item in data:
            sent_words = item[0]
            if self.pos is True and self.ner is True:
                instance = Instance(
                    words=sent_words, pos_tags=item[1], ner=item[2])
            elif self.pos is True:
                instance = Instance(words=sent_words, pos_tags=item[1])
            elif self.ner is True:
                instance = Instance(words=sent_words, ner=item[1])
            else:
                instance = Instance(words=sent_words)
            data_set.append(instance)
        data_set.apply(lambda ins: len(ins[Const.INPUT]), new_field_name=Const.INPUT_LEN)
        return data_set

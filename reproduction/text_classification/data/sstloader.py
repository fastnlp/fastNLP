from typing import Iterable
from nltk import Tree
from fastNLP.io.data_bundle import DataBundle, DataSetLoader
from fastNLP.core.vocabulary import VocabularyOption, Vocabulary
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP.io.embed_loader import EmbeddingOption, EmbedLoader
import csv
from typing import Union, Dict
from reproduction.utils import check_dataloader_paths, get_tokenizer


class SSTLoader(DataSetLoader):
    """
    读取SST数据集, DataSet包含fields::
        words: list(str) 需要分类的文本
        target: str 文本的标签
    数据来源: https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
    :param subtree: 是否将数据展开为子树，扩充数据量. Default: ``False``
    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    """

    URL = 'https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    DATA_DIR = 'sst/'

    def __init__(self, subtree=False, fine_grained=False):
        self.subtree = subtree
        tag_v = {'0': 'very negative', '1': 'negative', '2': 'neutral',
                 '3': 'positive', '4': 'very positive'}
        if not fine_grained:
            tag_v['0'] = tag_v['1']
            tag_v['4'] = tag_v['3']
        self.tag_v = tag_v

    def _load(self, path):
        """
        :param str path: 存储数据的路径
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            datas = []
            for l in f:
                datas.extend([(s, self.tag_v[t])
                              for s, t in self._get_one(l, self.subtree)])
        ds = DataSet()
        for words, tag in datas:
            ds.append(Instance(words=words, target=tag))
        return ds


    @staticmethod
    def _get_one(data, subtree):
        tree = Tree.fromstring(data)
        if subtree:
            return [(t.leaves(), t.label()) for t in tree.subtrees()]
        return [(tree.leaves(), tree.label())]


    def process(self,
                paths,
                train_ds: Iterable[str] = None,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,
                src_embed_op: EmbeddingOption = None):
        input_name, target_name = 'words', 'target'
        src_vocab = Vocabulary() if src_vocab_op is None else Vocabulary(**src_vocab_op)
        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_op is None else Vocabulary(**tgt_vocab_op)

        info = DataBundle(datasets=self.load(paths))
        _train_ds = [info.datasets[name]
                     for name in train_ds] if train_ds else info.datasets.values()
        src_vocab.from_dataset(*_train_ds, field_name=input_name)
        tgt_vocab.from_dataset(*_train_ds, field_name=target_name)
        src_vocab.index_dataset(
            *info.datasets.values(),
            field_name=input_name, new_field_name=input_name)
        tgt_vocab.index_dataset(
            *info.datasets.values(),
            field_name=target_name, new_field_name=target_name)
        info.vocabs = {
            input_name: src_vocab,
            target_name: tgt_vocab
        }


        if src_embed_op is not None:
            src_embed_op.vocab = src_vocab
            init_emb = EmbedLoader.load_with_vocab(**src_embed_op)
            info.embeddings[input_name] = init_emb


        for name, dataset in info.datasets.items():
            dataset.set_input(input_name)
            dataset.set_target(target_name)
        return info



class sst2Loader(DataSetLoader):
    '''
    数据来源"SST":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
    '''

    def __init__(self):
        super(sst2Loader, self).__init__()
        self.tokenizer = get_tokenizer()


    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        all_count=0
        csv_reader = csv.reader(open(path, encoding='utf-8'),delimiter='\t')
        skip_row = 0
        for idx,row in enumerate(csv_reader):
            if idx<=skip_row:
                continue
            target = row[1]
            words=self.tokenizer(row[0])
            ds.append(Instance(words=words,target=target))
            all_count+=1
        print("all count:", all_count)
        return ds



    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,
                src_embed_opt: EmbeddingOption = None,
                char_level_op=False):

        paths = check_dataloader_paths(paths)
        datasets = {}
        info = DataBundle()
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset

        def wordtochar(words):
            chars = []
            for word in words:
                word = word.lower()
                for char in word:
                    chars.append(char)
                chars.append('')
            chars.pop()
            return chars

        input_name, target_name = 'words', 'target'
        info.vocabs={}

        # 就分隔为char形式
        if char_level_op:
            for dataset in datasets.values():
                dataset.apply_field(wordtochar, field_name="words", new_field_name='chars')
        src_vocab = Vocabulary() if src_vocab_opt is None else Vocabulary(**src_vocab_opt)
        src_vocab.from_dataset(datasets['train'], field_name='words')
        src_vocab.index_dataset(*datasets.values(), field_name='words')

        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_opt is None else Vocabulary(**tgt_vocab_opt)
        tgt_vocab.from_dataset(datasets['train'], field_name='target')
        tgt_vocab.index_dataset(*datasets.values(), field_name='target')


        info.vocabs = {
            "words": src_vocab,
            "target": tgt_vocab
        }

        info.datasets = datasets

        if src_embed_opt is not None:
            embed = EmbedLoader.load_with_vocab(**src_embed_opt, vocab=src_vocab)
            info.embeddings['words'] = embed

        for name, dataset in info.datasets.items():
            dataset.set_input("words")
            dataset.set_target("target")

        return info



if __name__=="__main__":
    datapath = {"train": "/remote-home/ygwang/workspace/GLUE/SST-2/train.tsv",
                "dev": "/remote-home/ygwang/workspace/GLUE/SST-2/dev.tsv"}
    datainfo=sst2Loader().process(datapath,char_level_op=True)
    #print(datainfo.datasets["train"])

    len_count = 0
    for instance in datainfo.datasets["train"]:
        len_count += len(instance["chars"])
    ave_len = len_count / len(datainfo.datasets["train"])
    print(ave_len)
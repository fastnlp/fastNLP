import csv
from typing import Iterable
from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io.base_loader import DataInfo,DataSetLoader
from fastNLP.io.embed_loader import EmbeddingOption
from fastNLP.io.file_reader import _read_json
from typing import Union, Dict
from reproduction.Star_transformer.datasets import EmbedLoader
from reproduction.utils import check_dataloader_paths

class sst2Loader(DataSetLoader):
    '''
    数据来源"SST":'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8',
    '''
    def __init__(self):
        super(sst2Loader, self).__init__()

    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        all_count=0
        csv_reader = csv.reader(open(path, encoding='utf-8'),delimiter='\t')
        skip_row = 0
        for idx,row in enumerate(csv_reader):
            if idx<=skip_row:
                continue
            target = row[1]
            words = row[0].split()
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
        info = DataInfo()
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset

        def wordtochar(words):
            chars=[]
            for word in words:
                word=word.lower()
                for char in word:
                    chars.append(char)
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
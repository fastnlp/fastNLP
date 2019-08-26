import ast
import csv
from typing import Iterable
from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io import JsonLoader
from fastNLP.io.data_bundle import DataBundle,DataSetLoader
from fastNLP.io.embed_loader import EmbeddingOption
from fastNLP.io.file_reader import _read_json
from typing import Union, Dict
from reproduction.utils import check_dataloader_paths, get_tokenizer

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


class yelpLoader(DataSetLoader):
    
    """
    读取Yelp_full/Yelp_polarity数据集, DataSet包含fields:
        words: list(str), 需要分类的文本
        target: str, 文本的标签
        chars:list(str),未index的字符列表

    数据集：yelp_full/yelp_polarity
    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    """
    
    def __init__(self, fine_grained=False,lower=False):
        super(yelpLoader, self).__init__()
        tag_v = {'1.0': 'very negative', '2.0': 'negative', '3.0': 'neutral',
                 '4.0': 'positive', '5.0': 'very positive'}
        if not fine_grained:
            tag_v['1.0'] = tag_v['2.0']
            tag_v['5.0'] = tag_v['4.0']
        self.fine_grained = fine_grained
        self.tag_v = tag_v
        self.lower = lower
        self.tokenizer = get_tokenizer()

    '''
    读取Yelp数据集, DataSet包含fields:
    
        review_id: str, 22 character unique review id
        user_id: str, 22 character unique user id
        business_id: str, 22 character business id
        useful: int, number of useful votes received
        funny: int, number of funny votes received
        cool: int, number of cool votes received
        date: str, date formatted YYYY-MM-DD
        words: list(str), 需要分类的文本
        target: str, 文本的标签
    
    数据来源: https://www.yelp.com/dataset/download
    

    def _load_json(self, path):
        ds = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            d = ast.literal_eval(d)
            d["words"] = d.pop("text").split()
            d["target"] = self.tag_v[str(d.pop("stars"))]
            ds.append(Instance(**d))
        return ds
    
    def _load_yelp2015_broken(self,path):
        ds = DataSet()
        with open (path,encoding='ISO 8859-1') as f:
            row=f.readline()
            all_count=0
            exp_count=0
            while row:
                row=row.split("\t\t")
                all_count+=1
                if len(row)>=3:
                    words=row[-1].split()
                    try:
                        target=self.tag_v[str(row[-2])+".0"]
                        ds.append(Instance(words=words, target=target))
                    except KeyError:
                        exp_count+=1
                else:
                    exp_count+=1
                row = f.readline()
            print("error sample count:",exp_count)
            print("all count:",all_count)
        return ds
    '''

    def _load(self, path):
        ds = DataSet()
        csv_reader=csv.reader(open(path,encoding='utf-8'))
        all_count=0
        real_count=0
        for row in csv_reader:
            all_count+=1
            if len(row)==2:
                target=self.tag_v[row[0]+".0"]
                words = clean_str(row[1], self.tokenizer, self.lower)
                if len(words)!=0:
                    ds.append(Instance(words=words,target=target))
                    real_count += 1
        print("all count:", all_count)
        print("real count:", real_count)
        return ds



    def process(self, paths: Union[str, Dict[str, str]],
                train_ds: Iterable[str] = None,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,
                embed_opt: EmbeddingOption = None,
                char_level_op=False,
                split_dev_op=True
                ):
        paths = check_dataloader_paths(paths)
        datasets = {}
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

        input_name, target_name = 'words', 'target'
        info.vocabs={}
        #就分隔为char形式
        if char_level_op:
            for dataset in info.datasets.values():
                dataset.apply_field(wordtochar, field_name="words",new_field_name='chars')
        # if embed_opt is not None:
        #     embed = EmbedLoader.load_with_vocab(**embed_opt, vocab=vocab)
        #     info.embeddings['words'] = embed
        else:
            src_vocab.from_dataset(*_train_ds, field_name=input_name)
            src_vocab.index_dataset(*info.datasets.values(),field_name=input_name, new_field_name=input_name)
            info.vocabs[input_name]=src_vocab

        tgt_vocab.from_dataset(*_train_ds, field_name=target_name)
        tgt_vocab.index_dataset(
            *info.datasets.values(),
            field_name=target_name, new_field_name=target_name)

        info.vocabs[target_name]=tgt_vocab

        if split_dev_op:
            info.datasets['train'], info.datasets['dev'] = info.datasets['train'].split(0.1, shuffle=False)

        for name, dataset in info.datasets.items():
            dataset.set_input("words")
            dataset.set_target("target")

        return info

if __name__=="__main__":
    testloader=yelpLoader()
    # datapath = {"train": "/remote-home/ygwang/yelp_full/train.csv",
    #             "test": "/remote-home/ygwang/yelp_full/test.csv"}
    #datapath={"train": "/remote-home/ygwang/yelp_full/test.csv"}
    datapath = {"train": "/remote-home/ygwang/yelp_polarity/train.csv",
                "test": "/remote-home/ygwang/yelp_polarity/test.csv"}
    datainfo=testloader.process(datapath,char_level_op=True)

    len_count=0
    for instance in datainfo.datasets["train"]:
        len_count+=len(instance["chars"])

    ave_len=len_count/len(datainfo.datasets["train"])
    print(ave_len)


from fastNLP.io.embed_loader import EmbeddingOption, EmbedLoader
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io.base_loader import DataSetLoader, DataInfo
from typing import Union, Dict, List, Iterator
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const
from reproduction.utils import check_dataloader_paths
from functools import partial

class SigHanLoader(DataSetLoader):
    """
        任务相关的说明可以在这里找到http://sighan.cs.uchicago.edu/
        支持的数据格式为，一行一句，不同的word用空格隔开。如下例

            共同  创造  美好  的  新  世纪  ——  二○○一年  新年
            女士  们  ，  先生  们  ，  同志  们  ，  朋友  们  ：

        读取sighan中的数据集，返回的DataSet将包含以下的内容fields:
            raw_chars: list(str), 每个元素是一个汉字
            chars: list(str), 每个元素是一个index(汉字对应的index)
            target: list(int), 根据不同的encoding_type会有不同的变化

        :param target_type: target的类型，当前支持以下的两种: "bmes", "shift_relay"
    """

    def __init__(self, target_type:str):
        super().__init__()

        if target_type.lower() not in ('bmes', 'shift_relay'):
            raise ValueError("target_type only supports 'bmes', 'shift_relay'.")

        self.target_type = target_type
        if target_type=='bmes':
            self._word_len_to_target = self._word_len_to_bems
        elif target_type=='shift_relay':
            self._word_len_to_target = self._word_lens_to_relay

    @staticmethod
    def _word_lens_to_relay(word_lens: Iterator[int]):
        """
        [1, 2, 3, ..] 转换为[0, 1, 0, 2, 1, 0,](start指示seg有多长);
        :param word_lens:
        :return: {'target': , 'end_seg_mask':, 'start_seg_mask':}
        """
        tags = []
        end_seg_mask = []
        start_seg_mask = []
        for word_len in word_lens:
            tags.extend([idx for idx in range(word_len - 1, -1, -1)])
            end_seg_mask.extend([0] * (word_len - 1) + [1])
            start_seg_mask.extend([1] + [0] * (word_len - 1))
        return {'target': tags, 'end_seg_mask': end_seg_mask, 'start_seg_mask': start_seg_mask}

    @staticmethod
    def _word_len_to_bems(word_lens:Iterator[int])->Dict[str, List[str]]:
        """

        :param word_lens: 每个word的长度
        :return:
        """
        tags = []
        for word_len in word_lens:
            if word_len==1:
                tags.append('S')
            else:
                tags.append('B')
                for _ in range(word_len-2):
                    tags.append('M')
                tags.append('E')
        return {'target':tags}

    @staticmethod
    def _gen_bigram(chars:List[str])->List[str]:
        """

        :param chars:
        :return:
        """
        return [c1+c2 for c1, c2 in zip(chars, chars[1:]+['<eos>'])]

    def load(self, path:str, bigram:bool=False)->DataSet:
        """
        :param path: str
        :param bigram: 是否使用bigram feature
        :return:
        """
        dataset = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 去掉空行
                    continue
                parts = line.split()
                word_lens = map(len, parts)
                chars = list(''.join(parts))
                tags = self._word_len_to_target(word_lens)
                assert len(chars)==len(tags['target'])
                dataset.append(Instance(raw_chars=chars, **tags, seq_len=len(chars)))
        if len(dataset)==0:
            raise RuntimeError(f"{path} has no valid data.")
        if bigram:
            dataset.apply_field(self._gen_bigram, field_name='raw_chars', new_field_name='bigrams')
        return dataset

    def process(self, paths: Union[str, Dict[str, str]], char_vocab_opt:VocabularyOption=None,
                char_embed_opt:EmbeddingOption=None, bigram_vocab_opt:VocabularyOption=None,
                bigram_embed_opt:EmbeddingOption=None, L:int=4):
        """
        支持的数据格式为一行一个sample，并且用空格隔开不同的词语。例如

        Option::

            共同  创造  美好  的  新  世纪  ——  二○○一年  新年  贺词
            （  二○○○年  十二月  三十一日  ）  （  附  图片  1  张  ）
            女士  们  ，  先生  们  ，  同志  们  ，  朋友  们  ：

        paths支持两种格式，第一种是str，第二种是Dict[str, str].

        Option::

            # 1. str类型
            # 1.1 传入具体的文件路径
            data = SigHanLoader('bmes').process('/path/to/cws/data.txt') # 将读取data.txt的内容
            # 包含以下的内容data.vocabs['chars']:Vocabulary对象，
            #             data.vocabs['target']: Vocabulary对象，根据encoding_type可能会没有该值
            #             data.embeddings['chars']: Embedding对象. 只有提供了预训练的词向量的路径才有该项
            #             data.datasets['train']: DataSet对象
            #                   包含的field有:
            #                       raw_chars: list[str], 每个元素是一个汉字
            #                       chars: list[int], 每个元素是汉字对应的index
            #                       target: list[int], 根据encoding_type有对应的变化
            # 1.2 传入一个目录, 里面必须包含train.txt文件
            data = SigHanLoader('bmes').process('path/to/cws/') #将尝试在该目录下读取 train.txt, test.txt以及dev.txt
            # 包含以下的内容data.vocabs['chars']: Vocabulary对象
            #             data.vocabs['target']:Vocabulary对象
            #             data.embeddings['chars']: 仅在提供了预训练embedding路径的情况下，为Embedding对象;
            #             data.datasets['train']: DataSet对象
            #                    包含的field有:
            #                       raw_chars: list[str], 每个元素是一个汉字
            #                       chars: list[int], 每个元素是汉字对应的index
            #                       target: list[int], 根据encoding_type有对应的变化
            #             data.datasets['dev']: DataSet对象，如果文件夹下包含了dev.txt；内容与data.datasets['train']一样

            # 2. dict类型, key是文件的名称，value是对应的读取路径. 必须包含'train'这个key
            paths = {'train': '/path/to/train/train.txt', 'test':'/path/to/test/test.txt', 'dev':'/path/to/dev/dev.txt'}
            data = SigHanLoader(paths).process(paths)
            # 结果与传入目录时是一致的，但是可以传入多个数据集。data.datasets中的key将与这里传入的一致

        :param paths: 支持传入目录，文件路径，以及dict。
        :param char_vocab_opt: 用于构建chars的vocabulary参数，默认为min_freq=2
        :param char_embed_opt: 用于读取chars的Embedding的参数，默认不读取pretrained的embedding
        :param bigram_vocab_opt: 用于构建bigram的vocabulary参数，默认不使用bigram, 仅在指定该参数的情况下会带有bigrams这个field。
            为List[int], 每个instance长度与chars一样, abcde的bigram为ab bc cd de e<eos>
        :param bigram_embed_opt: 用于读取预训练bigram的参数，仅在传入bigram_vocab_opt有效
        :param L: 当target_type为shift_relay时传入的segment长度
        :return:
        """
        # 推荐大家使用这个check_data_loader_paths进行paths的验证
        paths = check_dataloader_paths(paths)
        datasets = {}
        data = DataInfo()
        bigram = bigram_vocab_opt is not None
        for name, path in paths.items():
            dataset = self.load(path, bigram=bigram)
            datasets[name] = dataset
        input_fields = []
        target_fields = []
        # 创建vocab
        char_vocab = Vocabulary(min_freq=2) if char_vocab_opt is None else Vocabulary(**char_vocab_opt)
        char_vocab.from_dataset(datasets['train'], field_name='raw_chars')
        char_vocab.index_dataset(*datasets.values(), field_name='raw_chars', new_field_name='chars')
        data.vocabs[Const.CHAR_INPUT] = char_vocab
        input_fields.extend([Const.CHAR_INPUT, Const.INPUT_LEN, Const.TARGET])
        target_fields.append(Const.TARGET)
        # 创建target
        if self.target_type == 'bmes':
            target_vocab = Vocabulary(unknown=None, padding=None)
            target_vocab.add_word_lst(['B']*4+['M']*3+['E']*2+['S'])
            target_vocab.index_dataset(*datasets.values(), field_name='target')
            data.vocabs[Const.TARGET] = target_vocab
        if char_embed_opt is not None:
            char_embed = EmbedLoader.load_with_vocab(**char_embed_opt, vocab=char_vocab)
            data.embeddings['chars'] = char_embed
        if bigram:
            bigram_vocab = Vocabulary(**bigram_vocab_opt)
            bigram_vocab.from_dataset(datasets['train'], field_name='bigrams')
            bigram_vocab.index_dataset(*datasets.values(), field_name='bigrams')
            data.vocabs['bigrams'] = bigram_vocab
            if bigram_embed_opt is not None:
                bigram_embed = EmbedLoader.load_with_vocab(**bigram_embed_opt, vocab=bigram_vocab)
                data.embeddings['bigrams'] = bigram_embed
            input_fields.append('bigrams')
        if self.target_type == 'shift_relay':
            func = partial(self._clip_target, L=L)
            for name, dataset in datasets.items():
                res = dataset.apply_field(func, field_name='target')
                relay_target = [res_i[0] for res_i in res]
                relay_mask = [res_i[1] for res_i in res]
                dataset.add_field('relay_target', relay_target, is_input=True, is_target=False, ignore_type=False)
                dataset.add_field('relay_mask', relay_mask, is_input=True, is_target=False, ignore_type=False)
        if self.target_type == 'shift_relay':
            input_fields.extend(['end_seg_mask'])
            target_fields.append('start_seg_mask')
        # 将dataset加入DataInfo
        for name, dataset in datasets.items():
            dataset.set_input(*input_fields)
            dataset.set_target(*target_fields)
            data.datasets[name] = dataset

        return data

    @staticmethod
    def _clip_target(target:List[int], L:int):
        """

        只有在target_type为shift_relay的使用
        :param target: List[int]
        :param L:
        :return:
        """
        relay_target_i = []
        tmp = []
        for j in range(len(target) - 1):
            tmp.append(target[j])
            if target[j] > target[j + 1]:
                pass
            else:
                relay_target_i.extend([L - 1 if t >= L else t for t in tmp[::-1]])
                tmp = []
        # 处理未结束的部分
        if len(tmp) == 0:
            relay_target_i.append(0)
        else:
            tmp.append(target[-1])
            relay_target_i.extend([L - 1 if t >= L else t for t in tmp[::-1]])
        relay_mask_i = []
        j = 0
        while j < len(target):
            seg_len = target[j] + 1
            if target[j] < L:
                relay_mask_i.extend([0] * (seg_len))
            else:
                relay_mask_i.extend([1] * (seg_len - L) + [0] * L)
            j = seg_len + j
        return relay_target_i, relay_mask_i


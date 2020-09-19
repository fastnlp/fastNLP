r"""
.. todo::
    doc
"""

__all__ = [
    "StaticEmbedding"
]
import os
import warnings
from collections import defaultdict
from copy import deepcopy
import json
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from ..core import logger
from ..core.vocabulary import Vocabulary
from ..io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
from ..io.file_utils import _get_file_name_base_on_postfix


VOCAB_FILENAME = 'vocab.txt'
STATIC_HYPER_FILENAME = 'static_hyper.json'
STATIC_EMBED_FILENAME = 'static.txt'


class StaticEmbedding(TokenEmbedding):
    r"""
    StaticEmbedding组件. 给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据(只会将出现在vocab中的词抽取出来，
    如果没有找到，则会随机初始化一个值(但如果该word是被标记为no_create_entry的话，则不会单独创建一个值，而是会被指向unk的index))。
    当前支持自动下载的预训练vector有:
    
    .. code::
    
        en: 实际为en-glove-840b-300d(常用)
        en-glove-6b-50d: glove官方的50d向量
        en-glove-6b-100d: glove官方的100d向量
        en-glove-6b-200d: glove官方的200d向量
        en-glove-6b-300d: glove官方的300d向量
        en-glove-42b-300d: glove官方使用42B数据训练版本
        en-glove-840b-300d:
        en-glove-twitter-27b-25d:
        en-glove-twitter-27b-50d:
        en-glove-twitter-27b-100d:
        en-glove-twitter-27b-200d:
        en-word2vec-300d: word2vec官方发布的300d向量
        en-fasttext-crawl: fasttext官方发布的300d英文预训练
        cn-char-fastnlp-100d: fastNLP训练的100d的character embedding
        cn-bi-fastnlp-100d: fastNLP训练的100d的bigram embedding
        cn-tri-fastnlp-100d: fastNLP训练的100d的trigram embedding
        cn-fasttext: fasttext官方发布的300d中文预训练embedding

    Example::
        
        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-50d')

        >>> vocab = Vocabulary().add_word_lst(["The", 'the', "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name="en-glove-50d", lower=True)
        >>> # "the", "The", "THE"它们共用一个vector，且将使用"the"在预训练词表中寻找它们的初始化表示。

        >>> vocab = Vocabulary().add_word_lst(["The", "the", "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE"]]])
        >>> embed(words)
        >>> tensor([[[ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849]]],
                   grad_fn=<EmbeddingBackward>)  # 每种word的输出是一致的。

    """
    
    def __init__(self, vocab: Vocabulary, model_dir_or_name: Union[str, None] = 'en', embedding_dim=-1, requires_grad: bool = True,
                 init_method=None, lower=False, dropout=0, word_dropout=0, normalize=False, min_freq=1, **kwargs):
        r"""
        
        :param vocab: Vocabulary. 若该项为None则会读取所有的embedding。
        :param model_dir_or_name: 可以有两种方式调用预训练好的static embedding：第一种是传入embedding文件夹(文件夹下应该只有一个
            以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，没有的话将自动下载。
            如果输入为None则使用embedding_dim的维度随机初始化一个embedding。
        :param int embedding_dim: 随机初始化的embedding的维度，当该值为大于0的值时，将忽略model_dir_or_name。
        :param bool requires_grad: 是否需要gradient. 默认为True
        :param callable init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法, 传入的方法应该接受一个tensor，并
            inplace地修改其值。
        :param bool lower: 是否将vocab中的词语小写后再和预训练的词表进行匹配。如果你的词表中包含大写的词语，或者就是需要单独
            为大写的词语开辟一个vector表示，则将lower设置为False。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param bool normalize: 是否对vector进行normalize，使得每个vector的norm为1。
        :param int min_freq: Vocabulary词频数小于这个数量的word将被指向unk。
        :param dict kwargs:
                bool only_train_min_freq: 仅对train中的词语使用min_freq筛选;
                bool only_norm_found_vector: 默认为False, 是否仅对在预训练中找到的词语使用normalize;
                bool only_use_pretrain_word: 默认为False, 仅使用出现在pretrain词表中的词，如果该词没有在预训练的词表中出现则为unk。如果embedding不需要更新建议设置为True。
        """
        super(StaticEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        if embedding_dim > 0:
            if model_dir_or_name:
                logger.info(f"StaticEmbedding will ignore `model_dir_or_name`, and randomly initialize embedding with"
                              f" dimension {embedding_dim}. If you want to use pre-trained embedding, "
                              f"set `embedding_dim` to 0.")
            model_dir_or_name = None
        
        # 得到cache_path
        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
            # 检查是否存在
        elif os.path.isfile(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = _get_file_name_base_on_postfix(os.path.abspath(os.path.expanduser(model_dir_or_name)), '.txt')
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        kwargs['min_freq'] = min_freq
        kwargs['lower'] = lower
        # 根据min_freq缩小vocab
        truncate_vocab = (vocab.min_freq is None and min_freq > 1) or (vocab.min_freq and vocab.min_freq < min_freq)
        if truncate_vocab:
            truncated_vocab = deepcopy(vocab)
            truncated_vocab.min_freq = min_freq
            truncated_vocab.word2idx = None
            if lower:  # 如果有lower，将大小写的的freq需要同时考虑到
                lowered_word_count = defaultdict(int)
                for word, count in truncated_vocab.word_count.items():
                    lowered_word_count[word.lower()] += count
                for word in truncated_vocab.word_count.keys():
                    word_count = truncated_vocab.word_count[word]
                    if lowered_word_count[word.lower()] >= min_freq and word_count < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - word_count),
                                                     no_create_entry=truncated_vocab._is_word_no_create_entry(word))
            
            # 只限制在train里面的词语使用min_freq筛选
            if kwargs.get('only_train_min_freq', False) and model_dir_or_name is not None:
                for word in truncated_vocab.word_count.keys():
                    if truncated_vocab._is_word_no_create_entry(word) and truncated_vocab.word_count[word] < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - truncated_vocab.word_count[word]),
                                                     no_create_entry=True)
            truncated_vocab.build_vocab()
            truncated_words_to_words = torch.arange(len(vocab)).long()
            for word, index in vocab:
                truncated_words_to_words[index] = truncated_vocab.to_index(word)
            logger.info(f"{len(vocab) - len(truncated_vocab)} words have frequency less than {min_freq}.")
            vocab = truncated_vocab

        self.only_use_pretrain_word = kwargs.get('only_use_pretrain_word', False)
        self.only_norm_found_vector = kwargs.get('only_norm_found_vector', False)
        # 读取embedding
        if lower:
            lowered_vocab = Vocabulary(padding=vocab.padding, unknown=vocab.unknown)
            for word, index in vocab:
                if vocab._is_word_no_create_entry(word):
                    lowered_vocab.add_word(word.lower(), no_create_entry=True)
                else:
                    lowered_vocab.add_word(word.lower())  # 先加入需要创建entry的
            logger.info(f"All word in the vocab have been lowered. There are {len(vocab)} words, {len(lowered_vocab)} "
                  f"unique lowered words.")
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=lowered_vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            if lowered_vocab.unknown:
                unknown_idx = lowered_vocab.unknown_idx
            else:
                unknown_idx = embedding.size(0) - 1  # 否则是最后一个为unknow
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            words_to_words = torch.full((len(vocab),), fill_value=unknown_idx, dtype=torch.long).long()
            for word, index in vocab:
                if word not in lowered_vocab:
                    word = word.lower()
                    if word not in lowered_vocab and lowered_vocab._is_word_no_create_entry(word):
                        continue  # 如果不需要创建entry,已经默认unknown了
                words_to_words[index] = self.words_to_words[lowered_vocab.to_index(word)]
            self.register_buffer('words_to_words', words_to_words)
            self._word_unk_index = lowered_vocab.unknown_idx  # 替换一下unknown的index
        else:
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
        if not self.only_norm_found_vector and normalize:
            embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)
        
        if truncate_vocab:
            for i in range(len(truncated_words_to_words)):
                index_in_truncated_vocab = truncated_words_to_words[i]
                truncated_words_to_words[i] = self.words_to_words[index_in_truncated_vocab]
            del self.words_to_words
            self.register_buffer('words_to_words', truncated_words_to_words)
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.kwargs = kwargs

    @property
    def weight(self):
        return self.embedding.weight
    
    def _randomly_init_embed(self, num_embedding, embedding_dim, init_embed=None):
        r"""

        :param int num_embedding: embedding的entry的数量
        :param int embedding_dim: embedding的维度大小
        :param callable init_embed: 初始化方法
        :return: torch.FloatTensor
        """
        embed = torch.zeros(num_embedding, embedding_dim)
        
        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)
        
        return embed
    
    def _load_with_vocab(self, embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>',
                         error='ignore', init_method=None):
        r"""
        从embed_filepath这个预训练的词向量中抽取出vocab这个词表的词的embedding。EmbedLoader将自动判断embed_filepath是
        word2vec(第一行只有两个元素)还是glove格式的数据。

        :param str embed_filepath: 预训练的embedding的路径。
        :param vocab: 词表 :class:`~fastNLP.Vocabulary` 类型，读取出现在vocab中的词的embedding。
            没有出现在vocab中的词的embedding将通过找到的词的embedding的正态分布采样出来，以使得整个Embedding是同分布的。
        :param dtype: 读出的embedding的类型
        :param str padding: 词表中padding的token
        :param str unknown: 词表中unknown的token
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法。默认使用torch.nn.init.zeros_
        :return torch.tensor:  shape为 [len(vocab), dimension], dimension由pretrain的embedding决定。
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}  # index是word在vocab中的index，value是vector或None(如果在pretrain中没有找到该word)
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e
            logger.info("Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab)))
            if not self.only_use_pretrain_word:  # 如果只用pretrain中的值就不要为未找到的词创建entry了
                for word, index in vocab:
                    if index not in matrix and not vocab._is_word_no_create_entry(word):
                        if found_unknown:  # 如果有unkonwn，用unknown初始化
                            matrix[index] = matrix[vocab.unknown_idx]
                        else:
                            matrix[index] = None
            # matrix中代表是需要建立entry的词
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)
            
            if vocab.unknown is None:  # 创建一个专门的unknown
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)), dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer('words_to_words', torch.full((len(vocab), ), fill_value=unknown_idx, dtype=torch.long).long())
            index = 0
            for word, index_in_vocab in vocab:
                if index_in_vocab in matrix:
                    vec = matrix.get(index_in_vocab)
                    if vec is not None:  # 使用找到的vector, 如果为None说明需要训练
                        vectors[index] = vec
                    self.words_to_words[index_in_vocab] = index
                    index += 1

            return vectors
    
    def forward(self, words):
        r"""
        传入words的index

        :param words: torch.LongTensor, [batch_size, max_len]
        :return: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words

    def save(self, folder):
        """
        将embedding存储到folder下，之后可以通过使用load方法读取

        :param str folder: 会在该folder下生成三个文件, vocab.txt, static_embed_hyper.txt, static_embed_hyper.json.
            其中vocab.txt可以用Vocabulary通过load读取; embedding.txt按照word2vec的方式存储，以空格的方式隔开元素,
            第一行只有两个元素，剩下的行首先是word然后是各个维度的值; static_embed_hyper.json是StaticEmbedding的超参数
        :return:
        """
        os.makedirs(folder, exist_ok=True)

        vocab = self.get_word_vocab()
        vocab_fp = os.path.join(folder, VOCAB_FILENAME)
        vocab.save(vocab_fp)
        kwargs = self.kwargs.copy()
        kwargs['dropout'] = self.dropout_layer.p
        kwargs['word_dropout'] = self.word_dropout
        kwargs['requires_grad'] = self.requires_grad
        kwargs['only_norm_found_vector'] = False
        kwargs['only_use_pretrain_word'] = True

        with open(os.path.join(folder, STATIC_HYPER_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(kwargs, f, indent=2)

        with open(os.path.join(folder, STATIC_EMBED_FILENAME), 'w', encoding='utf-8') as f:
            f.write('{}\n'.format(' '*30))  # 留白之后再来填写
            word_count = 0
            saved_word = {}
            valid_word_count = 0
            for i in range(len(self.words_to_words)):
                word = vocab.to_word(i)
                if not vocab._is_word_no_create_entry(word):
                    word_count += 1
                    if kwargs['lower']:
                        word = word.lower()
                    if word in saved_word:
                        continue
                    saved_word[word] = 1
                    vec_i = self.words_to_words[i]
                    if vec_i==vocab.unknown_idx and i!=vocab.unknown_idx:
                        continue
                    vec = self.embedding.weight.data[vec_i].tolist()
                    vec_str = ' '.join(map(str, vec))
                    f.write(f'{word} {vec_str}\n')
                    valid_word_count += 1
            f.seek(0)
            f.write('{} {}'.format(valid_word_count, self.embedding_dim))
        logger.debug(f"StaticEmbedding has been saved to {folder}.")

    @classmethod
    def load(cls, folder):
        """

        :param str folder: 该folder下应该有以下三个文件vocab.txt, static_embed.txt, static_hyper.json
        :return:
        """
        for name in [VOCAB_FILENAME, STATIC_EMBED_FILENAME, STATIC_HYPER_FILENAME]:
            assert os.path.exists(os.path.join(folder, name)), f"{name} not found in {folder}."

        vocab = Vocabulary.load(os.path.join(folder, VOCAB_FILENAME))
        with open(os.path.join(folder, STATIC_HYPER_FILENAME), 'r', encoding='utf-8') as f:
            hyper = json.load(f)

        logger.info(f"Load StaticEmbedding from {folder}.")
        embed = cls(vocab=vocab, model_dir_or_name=os.path.join(folder, STATIC_EMBED_FILENAME), **hyper)
        return embed


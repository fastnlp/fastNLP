__all__ = [
    "EmbedLoader",
    "EmbeddingOption",
]

import logging
import os
from typing import Callable

import numpy as np

from fastNLP.core.utils.utils import Option
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.log import logger


class EmbeddingOption(Option):
    def __init__(self,
                 embed_filepath=None,
                 dtype=np.float32,
                 normalize=True,
                 error='ignore'):
        super().__init__(
            embed_filepath=embed_filepath,
            dtype=dtype,
            normalize=normalize,
            error=error
        )


class EmbedLoader:
    r"""
    用于读取预训练的 embedding, 读取结果可直接载入为模型参数。
    """
    
    def __init__(self):
        super(EmbedLoader, self).__init__()

    @staticmethod
    def load_with_vocab(embed_filepath: str, vocab, dtype=np.float32, padding: str='<pad>', unknown: str='<unk>', normalize: bool=True,
                        error: str='ignore', init_method: Callable=None):
        r"""
        从 ``embed_filepath`` 这个预训练的词向量中抽取出 ``vocab`` 这个词表的词的 embedding。 :class:`EmbedLoader` 将自动判断 ``embed_filepath``
        是 **word2vec** （第一行只有两个元素） 还是 **glove** 格式的数据。

        :param embed_filepath: 预训练的 embedding 的路径。
        :param vocab: 词表 :class:`~fastNLP.core.Vocabulary` 类型，读取出现在 ``vocab`` 中的词的 embedding。
            没有出现在 ``vocab`` 中的词的 embedding 将通过找到的词的 embedding 的 *正态分布* 采样出来，以使得整个 Embedding 是同分布的。
        :param dtype: 读出的 embedding 的类型
        :param padding: 词表中 *padding* 的 token
        :param unknown: 词表中 *unknown* 的 token
        :param normalize: 是否将每个 vector 归一化到 norm 为 1
        :param error: 可以为以下值之一： ``['ignore', 'strict']`` 。如果为  ``ignore`` ，错误将自动跳过；如果是 ``strict`` ，错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param init_method: 用于初始化 embedding 的函数。该函数接受一个 :class:`numpy.ndarray` 类型，返回 :class:`numpy.ndarray`。
        :return: 返回类型为 :class:`numpy.ndarray`，形状为 ``[len(vocab), dimension]``，其中 *dimension*由预训练的 embedding 决定。
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            hit_flags = np.zeros(len(vocab), dtype=bool)
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = np.random.randn(len(vocab), dim).astype(dtype)
            if init_method:
                matrix = init_method(matrix)
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
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim)
                        hit_flags[index] = True
                except Exception as e:
                    if error == 'ignore':
                        logger.warning("Error occurred at the {} line.".format(idx))
                    else:
                        logging.error("Error occurred at the {} line.".format(idx))
                        raise e
            total_hits = sum(hit_flags)
            logging.info("Found {} out of {} words in the pre-training embedding.".format(total_hits, len(vocab)))
            if init_method is None:
                found_vectors = matrix[hit_flags]
                if len(found_vectors) != 0:
                    mean = np.mean(found_vectors, axis=0, keepdims=True)
                    std = np.std(found_vectors, axis=0, keepdims=True)
                    unfound_vec_num = len(vocab) - total_hits
                    r_vecs = np.random.randn(unfound_vec_num, dim).astype(dtype) * std + mean
                    matrix[hit_flags == False] = r_vecs

            if normalize:
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            
            return matrix
    
    @staticmethod
    def load_without_vocab(embed_filepath: str, dtype=np.float32, padding: str='<pad>', unknown: str='<unk>', normalize: bool=True,
                           error: str='ignore'):
        r"""
        从 ``embed_filepath`` 中读取预训练的 word vector。根据预训练的词表读取 embedding 并生成一个对应的 :class:`~fastNLP.core.Vocabulary` 。

        :param embed_filepath: 预训练的 embedding 的路径。
        :param dtype: 读出的 embedding 的类型
        :param padding: 词表中的 *padding* 的 token。
        :param unknown: 词表中的 *unknown* 的 token。
        :param normalize: 是否将每个 vector 归一化到 norm 为 1
        :param error: 可以为以下值之一： ``['ignore', 'strict']`` 。如果为  ``ignore`` ，错误将自动跳过；如果是 ``strict`` ，错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :return: 返回两个结果，第一个返回值为 :class:`numpy.ndarray`，大小为 ``[词表大小+x, 词表维度]`` 。 ``词表大小+x`` 是由于最终的大小还取决于
            是否使用 ``padding``，以及 ``unknown`` 有没有在词表中找到对应的词。 第二个返回值为 :class:`~fastNLP.core.Vocabulary` 类型的词表，其中
            词的顺序与 embedding 的顺序是一一对应的。

        """
        vocab = Vocabulary(padding=padding, unknown=unknown)
        vec_dict = {}
        found_unknown = False
        found_pad = False
        
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            start = 1
            dim = -1
            if len(line.strip().split()) != 2:
                f.seek(0)
                start = 0
            for idx, line in enumerate(f, start=start):
                try:
                    parts = line.strip().split()
                    if dim == -1:
                        dim = len(parts) - 1
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    vec = np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim)
                    vec_dict[word] = vec
                    vocab.add_word(word)
                    if unknown is not None and unknown == word:
                        found_unknown = True
                    if padding is not None and padding == word:
                        found_pad = True
                except Exception as e:
                    if error == 'ignore':
                        logger.warning("Error occurred at the {} line.".format(idx))
                        pass
                    else:
                        logging.error("Error occurred at the {} line.".format(idx))
                        raise e
            if dim == -1:
                raise RuntimeError("{} is an empty file.".format(embed_filepath))
            matrix = np.random.randn(len(vocab), dim).astype(dtype)
            for key, vec in vec_dict.items():
                index = vocab.to_index(key)
                matrix[index] = vec

            if ((unknown is not None) and (not found_unknown)) or ((padding is not None) and (not found_pad)):
                start_idx = 0
                if padding is not None:
                    start_idx += 1
                if unknown is not None:
                    start_idx += 1

                mean = np.mean(matrix[start_idx:], axis=0, keepdims=True)
                std = np.std(matrix[start_idx:], axis=0, keepdims=True)
                if (unknown is not None) and (not found_unknown):
                    matrix[start_idx - 1] = np.random.randn(1, dim).astype(dtype) * std + mean
                if (padding is not None) and (not found_pad):
                    matrix[0] = np.random.randn(1, dim).astype(dtype) * std + mean
            
            if normalize:
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            
            return matrix, vocab

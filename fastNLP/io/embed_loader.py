r"""
.. todo::
    doc
"""
__all__ = [
    "EmbedLoader",
    "EmbeddingOption",
]

import logging
import os
import warnings

import numpy as np

from ..core.utils import Option
from ..core.vocabulary import Vocabulary


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
    用于读取预训练的embedding, 读取结果可直接载入为模型参数。
    """
    
    def __init__(self):
        super(EmbedLoader, self).__init__()

    @staticmethod
    def load_with_vocab(embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>', normalize=True,
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
        :param bool normalize: 是否将每个vector归一化到norm为1
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param callable init_method: 传入numpy.ndarray, 返回numpy.ndarray, 用以初始化embedding
        :return numpy.ndarray:  shape为 [len(vocab), dimension], dimension由pretrain的embedding决定。
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
                        warnings.warn("Error occurred at the {} line.".format(idx))
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
    def load_without_vocab(embed_filepath, dtype=np.float32, padding='<pad>', unknown='<unk>', normalize=True,
                           error='ignore'):
        r"""
        从embed_filepath中读取预训练的word vector。根据预训练的词表读取embedding并生成一个对应的Vocabulary。

        :param str embed_filepath: 预训练的embedding的路径。
        :param dtype: 读出的embedding的类型
        :param str padding: 词表中的padding的token. 并以此用做vocab的padding。
        :param str unknown: 词表中的unknown的token. 并以此用做vocab的unknown。
        :param bool normalize: 是否将每个vector归一化到norm为1
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。这里主要可能出错的地
            方在于词表有空行或者词表出现了维度不一致。
        :return (numpy.ndarray, Vocabulary): Embedding的shape是[词表大小+x, 词表维度], "词表大小+x"是由于最终的大小还取决与
            是否使用padding, 以及unknown有没有在词表中找到对应的词。 Vocabulary中的词的顺序与Embedding的顺序是一一对应的。

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
                        warnings.warn("Error occurred at the {} line.".format(idx))
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

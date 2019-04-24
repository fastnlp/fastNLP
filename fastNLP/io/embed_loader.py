"""
.. _embed-loader:

用于读取预训练的embedding, 读取结果可直接载入为模型参数
"""
import os

import numpy as np

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.base_loader import BaseLoader

import warnings

class EmbedLoader(BaseLoader):
    """docstring for EmbedLoader"""

    def __init__(self):
        super(EmbedLoader, self).__init__()

    @staticmethod
    def load_with_vocab(embed_filepath, vocab, dtype=np.float32, normalize=True, error='ignore'):
        """
        load pretraining embedding in {embed_file} based on words in vocab. Words in vocab but not in the pretraining
        embedding are initialized from a normal distribution which has the mean and std of the found words vectors.
        The embedding type is determined automatically, support glove and word2vec(the first line only has two elements).

        :param embed_filepath: str, where to read pretrain embedding
        :param vocab: Vocabulary.
        :param dtype: the dtype of the embedding matrix
        :param normalize: bool, whether to normalize each word vector so that every vector has norm 1.
        :param error: str, 'ignore', 'strict'; if 'ignore' errors will not raise. if strict, any bad format error will
            raise
        :return: np.ndarray() will have the same [len(vocab), dimension], dimension is determined by the pretrain
            embedding
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            hit_flags = np.zeros(len(vocab), dtype=bool)
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts)==2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts)-1
                f.seek(0)
            matrix = np.random.randn(len(vocab), dim).astype(dtype)
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    if parts[0] in vocab:
                        index = vocab.to_index(parts[0])
                        matrix[index] = np.fromstring(' '.join(parts[1:]), sep=' ', dtype=dtype, count=dim)
                        hit_flags[index] = True
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        print("Error occurred at the {} line.".format(idx))
                        raise e
            total_hits = sum(hit_flags)
            print("Found {} out of {} words in the pre-training embedding.".format(total_hits, len(vocab)))
            found_vectors = matrix[hit_flags]
            if len(found_vectors)!=0:
                mean = np.mean(found_vectors, axis=0, keepdims=True)
                std = np.std(found_vectors, axis=0, keepdims=True)
                unfound_vec_num = len(vocab) - total_hits
                r_vecs = np.random.randn(unfound_vec_num, dim).astype(dtype)*std + mean
                matrix[hit_flags==False] = r_vecs

            if normalize:
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

            return matrix

    @staticmethod
    def load_without_vocab(embed_filepath, dtype=np.float32, padding='<pad>', unknown='<unk>', normalize=True,
                            error='ignore'):
        """
        load pretraining embedding in {embed_file}. And construct a Vocabulary based on the pretraining embedding.
        The embedding type is determined automatically, support glove and word2vec(the first line only has two elements).

        :param embed_filepath: str, where to read pretrain embedding
        :param dtype: the dtype of the embedding matrix
        :param padding: the padding tag for vocabulary.
        :param unknown: the unknown tag for vocabulary.
        :param normalize: bool, whether to normalize each word vector so that every vector has norm 1.
        :param error: str, 'ignore', 'strict'; if 'ignore' errors will not raise. if strict, any bad format error will
            :raise
        :return: np.ndarray() is determined by the pretraining embeddings
                 Vocabulary: contain all pretraining words and two special tag[<pad>, <unk>]

        """
        vocab = Vocabulary(padding=padding, unknown=unknown)
        vec_dict = {}
        found_unknown = False
        found_pad = False

        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline()
            start = 1
            dim = -1
            if len(line.strip().split())!=2:
                f.seek(0)
                start = 0
            for idx, line in enumerate(f, start=start):
                try:
                    parts = line.strip().split()
                    word = parts[0]
                    if dim==-1:
                        dim = len(parts)-1
                    vec = np.fromstring(' '.join(parts[1:]), sep=' ', dtype=dtype, count=dim)
                    vec_dict[word] = vec
                    vocab.add_word(word)
                    if unknown is not None and unknown==word:
                        found_unknown = True
                    if found_pad is not None and padding==word:
                        found_pad = True
                except Exception as e:
                    if error=='ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                        pass
                    else:
                        print("Error occurred at the {} line.".format(idx))
                        raise e
            if dim==-1:
                raise RuntimeError("{} is an empty file.".format(embed_filepath))
            matrix = np.random.randn(len(vocab), dim).astype(dtype)
            if (unknown is not None and not found_unknown) or (padding is not None and not found_pad):
                start_idx = 0
                if padding is not None:
                    start_idx += 1
                if unknown is not None:
                    start_idx += 1

                mean = np.mean(matrix[start_idx:], axis=0, keepdims=True)
                std = np.std(matrix[start_idx:], axis=0, keepdims=True)
                if (unknown is not None and not found_unknown):
                    matrix[start_idx-1] = np.random.randn(1, dim).astype(dtype)*std + mean
                if (padding is not None and not found_pad):
                    matrix[0] = np.random.randn(1, dim).astype(dtype)*std + mean

            for key, vec in vec_dict.items():
                index = vocab.to_index(key)
                matrix[index] = vec

            if normalize:
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

            return matrix, vocab

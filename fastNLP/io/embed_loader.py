import os

import numpy as np
import torch

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.base_loader import BaseLoader

import warnings

class EmbedLoader(BaseLoader):
    """docstring for EmbedLoader"""

    def __init__(self):
        super(EmbedLoader, self).__init__()

    @staticmethod
    def _load_glove(emb_file):
        """Read file as a glove embedding

        file format:
            embeddings are split by line,
            for one embedding, word and numbers split by space
        Example::

        word_1 float_1 float_2 ... float_emb_dim
        word_2 float_1 float_2 ... float_emb_dim
        ...
        """
        emb = {}
        with open(emb_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = list(filter(lambda w: len(w) > 0, line.strip().split(' ')))
                if len(line) > 2:
                    emb[line[0]] = torch.Tensor(list(map(float, line[1:])))
        return emb

    @staticmethod
    def _load_pretrain(emb_file, emb_type):
        """Read txt data from embedding file and convert to np.array as pre-trained embedding

        :param str emb_file: the pre-trained embedding file path
        :param str emb_type: the pre-trained embedding data format
        :return: a dict of ``{str: np.array}``
        """
        if emb_type == 'glove':
            return EmbedLoader._load_glove(emb_file)
        else:
            raise Exception("embedding type {} not support yet".format(emb_type))

    @staticmethod
    def load_embedding(emb_dim, emb_file, emb_type, vocab):
        """Load the pre-trained embedding and combine with the given dictionary.

        :param int emb_dim: the dimension of the embedding. Should be the same as pre-trained embedding.
        :param str emb_file: the pre-trained embedding file path.
        :param str emb_type: the pre-trained embedding format, support glove now
        :param Vocabulary vocab: a mapping from word to index, can be provided by user or built from pre-trained embedding
        :return (embedding_tensor, vocab):
                embedding_tensor - Tensor of shape (len(word_dict), emb_dim);
                vocab - input vocab or vocab built by pre-train

        """
        pretrain = EmbedLoader._load_pretrain(emb_file, emb_type)
        if vocab is None:
            # build vocabulary from pre-trained embedding
            vocab = Vocabulary()
            for w in pretrain.keys():
                vocab.add(w)
        embedding_tensor = torch.randn(len(vocab), emb_dim)
        for w, v in pretrain.items():
            if len(v.shape) > 1 or emb_dim != v.shape[0]:
                raise ValueError(
                    "Pretrained embedding dim is {}. Dimension dismatched. Required {}".format(v.shape, (emb_dim,)))
            if vocab.has_word(w):
                embedding_tensor[vocab[w]] = v
        return embedding_tensor, vocab

    @staticmethod
    def parse_glove_line(line):
        line = line.split()
        if len(line) <= 2:
            raise RuntimeError("something goes wrong in parsing glove embedding")
        return line[0], line[1:]

    @staticmethod
    def str_list_2_vec(line):
        try:
            return torch.Tensor(list(map(float, line)))
        except Exception:
            raise RuntimeError("something goes wrong in parsing glove embedding")


    @staticmethod
    def fast_load_embedding(emb_dim, emb_file, vocab):
        """Fast load the pre-trained embedding and combine with the given dictionary.
        This loading method uses line-by-line operation.

        :param int emb_dim: the dimension of the embedding. Should be the same as pre-trained embedding.
        :param str emb_file: the pre-trained embedding file path.
        :param Vocabulary vocab: a mapping from word to index, can be provided by user or built from pre-trained embedding
        :return embedding_matrix: numpy.ndarray

        """
        if vocab is None:
            raise RuntimeError("You must provide a vocabulary.")
        embedding_matrix = np.zeros(shape=(len(vocab), emb_dim), dtype=np.float32)
        hit_flags = np.zeros(shape=(len(vocab),), dtype=int)
        with open(emb_file, "r", encoding="utf-8") as f:
            startline = f.readline()
            if len(startline.split()) > 2:
                f.seek(0)
            for line in f:
                word, vector = EmbedLoader.parse_glove_line(line)
                if word in vocab:
                    vector = EmbedLoader.str_list_2_vec(vector)
                    if len(vector.shape) > 1 or emb_dim != vector.shape[0]:
                        raise ValueError("Pre-trained embedding dim is {}. Expect {}.".format(vector.shape, (emb_dim,)))
                    embedding_matrix[vocab[word]] = vector
                    hit_flags[vocab[word]] = 1

        if np.sum(hit_flags) < len(vocab):
            # some words from vocab are missing in pre-trained embedding
            # we normally sample each dimension
            vocab_embed = embedding_matrix[np.where(hit_flags)]
            sampled_vectors = np.random.normal(vocab_embed.mean(axis=0), vocab_embed.std(axis=0),
                                               size=(len(vocab) - np.sum(hit_flags), emb_dim))
            embedding_matrix[np.where(1 - hit_flags)] = sampled_vectors
        return embedding_matrix

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
                        raise e
            if dim==-1:
                raise RuntimeError("{} is an empty file.".format(embed_filepath))
            matrix = np.random.randn(len(vocab), dim).astype(dtype)
            # TODO 需要保证unk其它数据同分布的吗？
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

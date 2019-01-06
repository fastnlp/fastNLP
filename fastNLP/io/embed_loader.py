import numpy as np
import torch

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.base_loader import BaseLoader


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
        :return dict embedding: `{str: np.array}`
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
        :return embedding_tensor: Tensor of shape (len(word_dict), emb_dim)
                vocab: input vocab or vocab built by pre-train

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
        :return numpy.ndarray embedding_matrix:

        """
        if vocab is None:
            raise RuntimeError("You must provide a vocabulary.")
        embedding_matrix = np.zeros(shape=(len(vocab), emb_dim))
        hit_flags = np.zeros(shape=(len(vocab),), dtype=int)
        with open(emb_file, "r", encoding="utf-8") as f:
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

import _pickle
import os

import torch

from fastNLP.loader.base_loader import BaseLoader
from fastNLP.core.vocabulary import Vocabulary


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
                line = list(filter(lambda w: len(w)>0, line.strip().split(' ')))
                if len(line) > 0:
                    emb[line[0]] = torch.Tensor(list(map(float, line[1:])))
        return emb
    
    @staticmethod
    def _load_pretrain(emb_file, emb_type):
        """Read txt data from embedding file and convert to np.array as pre-trained embedding

        :param emb_file: str, the pre-trained embedding file path
        :param emb_type: str, the pre-trained embedding data format
        :return dict: {str: np.array}
        """
        if emb_type == 'glove':
            return EmbedLoader._load_glove(emb_file)
        else:
            raise Exception("embedding type {} not support yet".format(emb_type))

    @staticmethod
    def load_embedding(emb_dim, emb_file, emb_type, vocab, emb_pkl):
        """Load the pre-trained embedding and combine with the given dictionary.

        :param emb_dim: int, the dimension of the embedding. Should be the same as pre-trained embedding.
        :param emb_file: str, the pre-trained embedding file path.
        :param emb_type: str, the pre-trained embedding format, support glove now
        :param vocab: Vocabulary, a mapping from word to index, can be provided by user or built from pre-trained embedding
        :param emb_pkl: str, the embedding pickle file.
        :return embedding_tensor: Tensor of shape (len(word_dict), emb_dim)
                vocab: input vocab or vocab built by pre-train
        TODO: fragile code
        """
        # If the embedding pickle exists, load it and return.
        if os.path.exists(emb_pkl):
            with open(emb_pkl, "rb") as f:
                embedding_tensor, vocab = _pickle.load(f)
            return embedding_tensor, vocab
        # Otherwise, load the pre-trained embedding.
        pretrain = EmbedLoader._load_pretrain(emb_file, emb_type)
        if vocab is None:
            # build vocabulary from pre-trained embedding
            vocab = Vocabulary()
            for w in pretrain.keys():
                vocab.update(w)
        embedding_tensor = torch.randn(len(vocab), emb_dim)
        for w, v in pretrain.items():
            if len(v.shape) > 1 or emb_dim != v.shape[0]:
                raise ValueError('pretrian embedding dim is {}, dismatching required {}'.format(v.shape, (emb_dim,)))
            if vocab.has_word(w):
                embedding_tensor[vocab[w]] = v

        # save and return the result
        with open(emb_pkl, "wb") as f:
            _pickle.dump((embedding_tensor, vocab), f)
        return embedding_tensor, vocab

import _pickle
import os

import numpy as np

from fastNLP.loader.base_loader import BaseLoader


class EmbedLoader(BaseLoader):
    """docstring for EmbedLoader"""

    def __init__(self, data_path):
        super(EmbedLoader, self).__init__(data_path)

    @staticmethod
    def load_embedding(emb_dim, emb_file, word_dict, emb_pkl):
        """Load the pre-trained embedding and combine with the given dictionary.

        :param emb_file: str, the pre-trained embedding.
                The embedding file should have the following format:
                    Each line is a word embedding, where a word string is followed by multiple floats.
                    Floats are separated by space. The word and the first float are separated by space.
        :param word_dict: dict, a mapping from word to index.
        :param emb_dim: int, the dimension of the embedding. Should be the same as pre-trained embedding.
        :param emb_pkl: str, the embedding pickle file.
        :return embedding_np: numpy array of shape (len(word_dict), emb_dim)

        TODO: fragile code
        """
        # If the embedding pickle exists, load it and return.
        if os.path.exists(emb_pkl):
            with open(emb_pkl, "rb") as f:
                embedding_np = _pickle.load(f)
            return embedding_np
        # Otherwise, load the pre-trained embedding.
        with open(emb_file, "r", encoding="utf-8") as f:
            # begin with a random embedding
            embedding_np = np.random.uniform(-1, 1, size=(len(word_dict), emb_dim))
            for line in f:
                line = line.strip().split()
                if len(line) != emb_dim + 1:
                    # skip this line if two embedding dimension not match
                    continue
                if line[0] in word_dict:
                    # find the word and replace its embedding with a pre-trained one
                    embedding_np[word_dict[line[0]]] = [float(i) for i in line[1:]]
        # save and return the result
        with open(emb_pkl, "wb") as f:
            _pickle.dump(embedding_np, f)
        return embedding_np

import os
import errno
import collections
import torch
import numpy as np
import pyhocon



# flatten the list
def flatten(l):
    return [item for sublist in l for item in sublist]


def get_config(filename):
    return pyhocon.ConfigFactory.parse_file(filename)


# safe make directions
def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = ["<unk>"]
    with open(char_vocab_path) as f:
        vocab.extend(c.strip() for c in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c: i for i, c in enumerate(vocab)})
    return char_dict

# 加载embedding
def load_embedding_dict(embedding_path, embedding_size, embedding_format):
    print("Loading word embeddings from {}...".format(embedding_path))
    default_embedding = np.zeros(embedding_size)
    embedding_dict = collections.defaultdict(lambda: default_embedding)
    skip_first = embedding_format == "vec"
    with open(embedding_path) as f:
        for i, line in enumerate(f.readlines()):
            if skip_first and i == 0:
                continue
            splits = line.split()
            assert len(splits) == embedding_size + 1
            word = splits[0]
            embedding = np.array([float(s) for s in splits[1:]])
            embedding_dict[word] = embedding
    print("Done loading word embeddings.")
    return embedding_dict


# safe devide
def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


def shape(x, dim):
    return x.get_shape()[dim].value or torch.shape(x)[dim]


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    else:
        return v


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1



if __name__=="__main__":
    print(load_char_dict("../data/char_vocab.english.txt"))
    embedding_dict = load_embedding_dict("../data/glove.840B.300d.txt.filtered",300,"txt")
    print("hello")

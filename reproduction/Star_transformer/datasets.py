import torch
import json
import os
from fastNLP import Vocabulary
from fastNLP.io.data_loader import ConllLoader, SSTLoader, SNLILoader
from fastNLP.core import Const as C
import numpy as np

MAX_LEN = 128

def update_v(vocab, data, field):
    data.apply(lambda x: vocab.add_word_lst(x[field]), new_field_name=None)


def to_index(vocab, data, field, name):
    def func(x):
        try:
            return [vocab.to_index(w) for w in x[field]]
        except ValueError:
            return [vocab.padding_idx for _ in x[field]]
    data.apply(func, new_field_name=name)


def load_seqtag(path, files, indexs):
    word_h, tag_h = 'words', 'tags'
    loader = ConllLoader(headers=[word_h, tag_h], indexes=indexs)
    ds_list = []
    for fn in files:
        ds_list.append(loader.load(os.path.join(path, fn)))
    word_v = Vocabulary(min_freq=2)
    tag_v = Vocabulary(unknown=None)
    update_v(word_v, ds_list[0], word_h)
    update_v(tag_v, ds_list[0], tag_h)

    def process_data(ds):
        to_index(word_v, ds, word_h, C.INPUT)
        to_index(tag_v, ds, tag_h, C.TARGET)
        ds.apply(lambda x: x[C.INPUT][:MAX_LEN], new_field_name=C.INPUT)
        ds.apply(lambda x: x[C.TARGET][:MAX_LEN], new_field_name=C.TARGET)
        ds.apply(lambda x: len(x[word_h]), new_field_name=C.INPUT_LEN)
        ds.set_input(C.INPUT, C.INPUT_LEN)
        ds.set_target(C.TARGET, C.INPUT_LEN)
    for i in range(len(ds_list)):
        process_data(ds_list[i])
    return ds_list, word_v, tag_v


def load_sst(path, files):
    loaders = [SSTLoader(subtree=sub, fine_grained=True)
               for sub in [True, False, False]]
    ds_list = [loader.load(os.path.join(path, fn))
               for fn, loader in zip(files, loaders)]
    word_v = Vocabulary(min_freq=0)
    tag_v = Vocabulary(unknown=None, padding=None)
    for ds in ds_list:
        ds.apply(lambda x: [w.lower()
                            for w in x['words']], new_field_name='words')
    #ds_list[0].drop(lambda x: len(x['words']) < 3)
    update_v(word_v, ds_list[0], 'words')
    update_v(word_v, ds_list[1], 'words')
    update_v(word_v, ds_list[2], 'words')
    ds_list[0].apply(lambda x: tag_v.add_word(
        x['target']), new_field_name=None)

    def process_data(ds):
        to_index(word_v, ds, 'words', C.INPUT)
        ds.apply(lambda x: tag_v.to_index(x['target']), new_field_name=C.TARGET)
        ds.apply(lambda x: x[C.INPUT][:MAX_LEN], new_field_name=C.INPUT)
        ds.apply(lambda x: len(x['words']), new_field_name=C.INPUT_LEN)
        ds.set_input(C.INPUT, C.INPUT_LEN)
        ds.set_target(C.TARGET)
    for i in range(len(ds_list)):
        process_data(ds_list[i])
    return ds_list, word_v, tag_v


def load_snli(path, files):
    loader = SNLILoader()
    ds_list = [loader.load(os.path.join(path, f)) for f in files]
    word_v = Vocabulary(min_freq=2)
    tag_v = Vocabulary(unknown=None, padding=None)
    for ds in ds_list:
        ds.apply(lambda x: [w.lower()
                            for w in x['words1']], new_field_name='words1')
        ds.apply(lambda x: [w.lower()
                            for w in x['words2']], new_field_name='words2')
    update_v(word_v, ds_list[0], 'words1')
    update_v(word_v, ds_list[0], 'words2')
    ds_list[0].apply(lambda x: tag_v.add_word(
        x['target']), new_field_name=None)

    def process_data(ds):
        to_index(word_v, ds, 'words1', C.INPUTS(0))
        to_index(word_v, ds, 'words2', C.INPUTS(1))
        ds.apply(lambda x: tag_v.to_index(x['target']), new_field_name=C.TARGET)
        ds.apply(lambda x: x[C.INPUTS(0)][:MAX_LEN], new_field_name=C.INPUTS(0))
        ds.apply(lambda x: x[C.INPUTS(1)][:MAX_LEN], new_field_name=C.INPUTS(1))
        ds.apply(lambda x: len(x[C.INPUTS(0)]), new_field_name=C.INPUT_LENS(0))
        ds.apply(lambda x: len(x[C.INPUTS(1)]), new_field_name=C.INPUT_LENS(1))
        ds.set_input(C.INPUTS(0), C.INPUTS(1), C.INPUT_LENS(0), C.INPUT_LENS(1))
        ds.set_target(C.TARGET)
    for i in range(len(ds_list)):
        process_data(ds_list[i])
    return ds_list, word_v, tag_v


class EmbedLoader:
    @staticmethod
    def parse_glove_line(line):
        line = line.split()
        if len(line) <= 2:
            raise RuntimeError(
                "something goes wrong in parsing glove embedding")
        return line[0], line[1:]

    @staticmethod
    def str_list_2_vec(line):
        return torch.Tensor(list(map(float, line)))

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
        embedding_matrix = np.zeros(
            shape=(len(vocab), emb_dim), dtype=np.float32)
        hit_flags = np.zeros(shape=(len(vocab),), dtype=int)
        with open(emb_file, "r", encoding="utf-8") as f:
            startline = f.readline()
            if len(startline.split()) > 2:
                f.seek(0)
            for line in f:
                word, vector = EmbedLoader.parse_glove_line(line)
                try:
                    if word in vocab:
                        vector = EmbedLoader.str_list_2_vec(vector)
                        if emb_dim != vector.size(0):
                            continue
                        embedding_matrix[vocab[word]] = vector
                        hit_flags[vocab[word]] = 1
                except Exception:
                    continue

        if np.sum(hit_flags) < len(vocab):
            # some words from vocab are missing in pre-trained embedding
            # we normally sample each dimension
            vocab_embed = embedding_matrix[np.where(hit_flags)]
            #sampled_vectors = np.random.normal(vocab_embed.mean(axis=0), vocab_embed.std(axis=0),
            #                                   size=(len(vocab) - np.sum(hit_flags), emb_dim))
            sampled_vectors = np.random.uniform(-0.01, 0.01,
                                               size=(len(vocab) - np.sum(hit_flags), emb_dim))

            embedding_matrix[np.where(1 - hit_flags)] = sampled_vectors
        return embedding_matrix

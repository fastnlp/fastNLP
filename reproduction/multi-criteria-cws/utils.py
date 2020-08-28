import numpy as np
import torch
import torch.cuda
import random
import os
import sys
import errno
import time
import codecs
import hashlib
import _pickle as pickle
import warnings
from fastNLP.io import EmbedLoader

UNK_TAG = "<unk>"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t.upper() == "B" or t.upper() == "S":
            result.append(word)
            word = ""
        word += c
    if len(word) != 0:
        result.append(word)

    return result


def bmes_to_index(tags):
    result = []
    if len(tags) == 0:
        return result
    word = (0, 0)

    for i, t in enumerate(tags):
        if i == 0:
            word = (0, 0)
        elif t.upper() == "B" or t.upper() == "S":
            result.append(word)
            word = (i, 0)
        word = (word[0], word[1] + 1)
    if word[1] != 0:
        result.append(word)
    return result


def get_bmes(sent):
    x = []
    y = []
    for word in sent:
        length = len(word)
        tag = ["m"] * length if length > 1 else ["s"] * length
        if length > 1:
            tag[0] = "b"
            tag[-1] = "e"
        x += list(word)
        y += tag
    return x, y


class CWSEvaluator:
    def __init__(self, i2t):
        self.correct_preds = 0.0
        self.total_preds = 0.0
        self.total_correct = 0.0
        self.i2t = i2t

    def add_instance(self, pred_tags, gold_tags):
        pred_tags = [self.i2t[i] for i in pred_tags]
        gold_tags = [self.i2t[i] for i in gold_tags]
        # Evaluate PRF
        lab_gold_chunks = set(bmes_to_index(gold_tags))
        lab_pred_chunks = set(bmes_to_index(pred_tags))
        self.correct_preds += len(lab_gold_chunks & lab_pred_chunks)
        self.total_preds += len(lab_pred_chunks)
        self.total_correct += len(lab_gold_chunks)

    def result(self, percentage=True):
        p = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        r = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        if percentage:
            p *= 100
            r *= 100
            f1 *= 100
        return p, r, f1


class CWS_OOV:
    def __init__(self, dic):
        self.dic = dic
        self.recall = 0
        self.tot = 0

    def update(self, gold_sent, pred_sent):
        i = 0
        j = 0
        id = 0
        for w in gold_sent:
            if w not in self.dic:
                self.tot += 1
                while i + len(pred_sent[id]) <= j:
                    i += len(pred_sent[id])
                    id += 1
                if (
                    i == j
                    and len(pred_sent[id]) == len(w)
                    and w.find(pred_sent[id]) != -1
                ):
                    self.recall += 1
            j += len(w)
        # print(gold_sent,pred_sent,self.tot)

    def oov(self, percentage=True):
        ins = 1.0 * self.recall / self.tot
        if percentage:
            ins *= 100
        return ins


def get_processing_word(
    vocab_words=None, vocab_chars=None, lowercase=False, chars=False
):
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = "0"

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK_TAG]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars:
            return char_ids, word
        else:
            return word

    return f


def append_tags(src, des, name, part, encode="utf-16"):
    with open("{}/{}.txt".format(src, part), encoding=encode) as input, open(
        "{}/{}.txt".format(des, part), "a", encoding=encode
    ) as output:
        for line in input:
            line = line.strip()
            if len(line) > 0:
                output.write("<{}> {} </{}>".format(name, line, name))
            output.write("\n")


def is_dataset_tag(word):
    return len(word) > 2 and word[0] == "<" and word[-1] == ">"


def to_tag_strings(i2ts, tag_mapping, pos_separate_col=True):
    senlen = len(tag_mapping)
    key_value_strs = []

    for j in range(senlen):
        val = i2ts[tag_mapping[j]]
        pos_str = val
        key_value_strs.append(pos_str)
    return key_value_strs


def to_id_list(w2i):
    i2w = [None] * len(w2i)
    for w, i in w2i.items():
        i2w[i] = w
    return i2w


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def md5_for_file(fn):
    md5 = hashlib.md5()
    with open(fn, "rb") as f:
        for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def embedding_match_vocab(
    vocab,
    emb,
    ori_vocab,
    dtype=np.float32,
    padding="<pad>",
    unknown="<unk>",
    normalize=True,
    error="ignore",
    init_method=None,
):
    dim = emb.shape[-1]
    matrix = np.random.randn(len(vocab), dim).astype(dtype)
    hit_flags = np.zeros(len(vocab), dtype=bool)

    if init_method:
        matrix = init_method(matrix)
    for word, idx in ori_vocab.word2idx.items():
        try:
            if word == padding and vocab.padding is not None:
                word = vocab.padding
            elif word == unknown and vocab.unknown is not None:
                word = vocab.unknown
            if word in vocab:
                index = vocab.to_index(word)
                matrix[index] = emb[idx]
                hit_flags[index] = True
        except Exception as e:
            if error == "ignore":
                warnings.warn("Error occurred at the {} line.".format(idx))
            else:
                print("Error occurred at the {} line.".format(idx))
                raise e

    total_hits = np.sum(hit_flags)
    print(
        "Found {} out of {} words in the pre-training embedding.".format(
            total_hits, len(vocab)
        )
    )
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


def embedding_load_with_cache(emb_file, cache_dir, vocab, **kwargs):
    def match_cache(file, cache_dir):
        md5 = md5_for_file(file)
        cache_files = os.listdir(cache_dir)
        for fn in cache_files:
            if md5 in fn.split("-")[-1]:
                return os.path.join(cache_dir, fn), True
        return (
            "{}-{}.pkl".format(os.path.join(cache_dir, os.path.basename(file)), md5),
            False,
        )

    def get_cache(file):
        if not os.path.exists(file):
            return None
        with open(file, "rb") as f:
            emb = pickle.load(f)
        return emb

    os.makedirs(cache_dir, exist_ok=True)
    cache_fn, match = match_cache(emb_file, cache_dir)
    if not match:
        print("cache missed, re-generating cache at {}".format(cache_fn))
        emb, ori_vocab = EmbedLoader.load_without_vocab(
            emb_file, padding=None, unknown=None, normalize=False
        )
        with open(cache_fn, "wb") as f:
            pickle.dump((emb, ori_vocab), f)

    else:
        print("cache matched at {}".format(cache_fn))

    # use cache
    print("loading embeddings ...")
    emb = get_cache(cache_fn)
    assert emb is not None
    return embedding_match_vocab(vocab, emb[0], emb[1], **kwargs)

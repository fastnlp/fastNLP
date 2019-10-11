import os
import sys

import codecs
import argparse
from _pickle import load, dump
import collections
from utils import get_processing_word, is_dataset_tag, make_sure_path_exists, get_bmes
from fastNLP import Instance, DataSet, Vocabulary, Const

max_len = 0


def expand(x):
    sent = ["<sos>"] + x[1:] + ["<eos>"]
    return [x + y for x, y in zip(sent[:-1], sent[1:])]


def read_file(filename, processing_word=get_processing_word(lowercase=False)):
    dataset = DataSet()
    niter = 0
    with codecs.open(filename, "r", "utf-8-sig") as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    assert len(words) > 2
                    if niter == 1:
                        print(words, tags)
                    niter += 1
                    dataset.append(Instance(ori_words=words[:-1], ori_tags=tags[:-1]))
                    words, tags = [], []
            else:
                word, tag = line.split()
                word = processing_word(word)
                words.append(word)
                tags.append(tag.lower())

    dataset.apply_field(lambda x: [x[0]], field_name="ori_words", new_field_name="task")
    dataset.apply_field(
        lambda x: len(x), field_name="ori_tags", new_field_name="seq_len"
    )
    dataset.apply_field(
        lambda x: expand(x), field_name="ori_words", new_field_name="bi1"
    )
    return dataset


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--data_path", required=True, type=str, help="all of datasets pkl paths")
    # fmt: on

    options, _ = parser.parse_known_args()

    train_set, test_set = DataSet(), DataSet()

    input_dir = os.path.join(options.data_path, "joint-sighan2008/bmes")
    options.output = os.path.join(options.data_path, "total_dataset.pkl")
    print(input_dir, options.output)

    for fn in os.listdir(input_dir):
        if fn not in ["test.txt", "train-all.txt"]:
            continue
        print(fn)
        abs_fn = os.path.join(input_dir, fn)
        ds = read_file(abs_fn)
        if "test.txt" == fn:
            test_set = ds
        else:
            train_set = ds

    print(
        "num samples of total train, test: {}, {}".format(len(train_set), len(test_set))
    )

    uni_vocab = Vocabulary(min_freq=None).from_dataset(
        train_set, test_set, field_name="ori_words"
    )
    # bi_vocab = Vocabulary(min_freq=3, max_size=50000).from_dataset(train_set,test_set, field_name="bi1")
    bi_vocab = Vocabulary(min_freq=3, max_size=None).from_dataset(
        train_set, field_name="bi1", no_create_entry_dataset=[test_set]
    )
    tag_vocab = Vocabulary(min_freq=None, padding="s", unknown=None).from_dataset(
        train_set, field_name="ori_tags"
    )
    task_vocab = Vocabulary(min_freq=None, padding=None, unknown=None).from_dataset(
        train_set, field_name="task"
    )

    def to_index(dataset):
        uni_vocab.index_dataset(dataset, field_name="ori_words", new_field_name="uni")
        tag_vocab.index_dataset(dataset, field_name="ori_tags", new_field_name="tags")
        task_vocab.index_dataset(dataset, field_name="task", new_field_name="task")

        dataset.apply_field(lambda x: x[1:], field_name="bi1", new_field_name="bi2")
        dataset.apply_field(lambda x: x[:-1], field_name="bi1", new_field_name="bi1")
        bi_vocab.index_dataset(dataset, field_name="bi1", new_field_name="bi1")
        bi_vocab.index_dataset(dataset, field_name="bi2", new_field_name="bi2")

        dataset.set_input("task", "uni", "bi1", "bi2", "seq_len")
        dataset.set_target("tags")
        return dataset

    train_set = to_index(train_set)
    test_set = to_index(test_set)

    output = {}
    output["train_set"] = train_set
    output["test_set"] = test_set
    output["uni_vocab"] = uni_vocab
    output["bi_vocab"] = bi_vocab
    output["tag_vocab"] = tag_vocab
    output["task_vocab"] = task_vocab

    print(tag_vocab.word2idx)
    print(task_vocab.word2idx)

    make_sure_path_exists(os.path.dirname(options.output))

    print("Saving dataset to {}".format(os.path.abspath(options.output)))
    with open(options.output, "wb") as outfile:
        dump(output, outfile)

    print(len(task_vocab), len(tag_vocab), len(uni_vocab), len(bi_vocab))
    dic = {}
    tokens = {}

    def process(words):
        name = words[0][1:-1]
        if name not in dic:
            dic[name] = set()
            tokens[name] = 0
        tokens[name] += len(words[1:])
        dic[name].update(words[1:])

    train_set.apply_field(process, "ori_words", None)
    for name in dic.keys():
        print(name, len(dic[name]), tokens[name])

    with open(os.path.join(os.path.dirname(options.output), "oovdict.pkl"), "wb") as f:
        dump(dic, f)

    def get_max_len(ds):
        global max_len
        max_len = 0

        def find_max_len(words):
            global max_len
            if max_len < len(words):
                max_len = len(words)

        ds.apply_field(find_max_len, "ori_words", None)
        return max_len

    print(
        "train max len: {}, test max len: {}".format(
            get_max_len(train_set), get_max_len(test_set)
        )
    )


if __name__ == "__main__":
    main()

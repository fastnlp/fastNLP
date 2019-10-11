import os
import re
import argparse
from opencc import OpenCC

cc = OpenCC("t2s")

from utils import make_sure_path_exists, append_tags

sighan05_root = ""
sighan08_root = ""
data_path = ""

E_pun = u",.!?[]()<>\"\"'',"
C_pun = u"，。！？【】（）《》“”‘’、"
Table = {ord(f): ord(t) for f, t in zip(C_pun, E_pun)}
Table[12288] = 32  # 全半角空格


def C_trans_to_E(string):
    return string.translate(Table)


def normalize(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def preprocess(text):
    rNUM = u"(-|\+)?\d+((\.|·)\d+)?%?"
    rENG = u"[A-Za-z_]+.*"
    sent = normalize(C_trans_to_E(text.strip())).split()
    new_sent = []
    for word in sent:
        word = re.sub(u"\s+", "", word, flags=re.U)
        word = re.sub(rNUM, u"0", word, flags=re.U)
        word = re.sub(rENG, u"X", word)
        new_sent.append(word)
    return new_sent


def to_sentence_list(text, split_long_sentence=False):
    text = preprocess(text)
    delimiter = set()
    delimiter.update("。！？：；…、，（）,;!?、,\"'")
    delimiter.add("……")
    sent_list = []
    sent = []
    sent_len = 0
    for word in text:
        sent.append(word)
        sent_len += len(word)
        if word in delimiter or (split_long_sentence and sent_len >= 50):
            sent_list.append(sent)
            sent = []
            sent_len = 0

    if len(sent) > 0:
        sent_list.append(sent)

    return sent_list


def is_traditional(dataset):
    return dataset in ["as", "cityu", "ckip"]


def convert_file(
    src, des, need_cc=False, split_long_sentence=False, encode="utf-8-sig"
):
    with open(src, encoding=encode) as src, open(des, "w", encoding="utf-8") as des:
        for line in src:
            for sent in to_sentence_list(line, split_long_sentence):
                line = " ".join(sent) + "\n"
                if need_cc:
                    line = cc.convert(line)
                des.write(line)
                # if len(''.join(sent)) > 200:
                #     print(' '.join(sent))


def split_train_dev(dataset):
    root = data_path + "/" + dataset + "/raw/"
    with open(root + "train-all.txt", encoding="UTF-8") as src, open(
        root + "train.txt", "w", encoding="UTF-8"
    ) as train, open(root + "dev.txt", "w", encoding="UTF-8") as dev:
        lines = src.readlines()
        idx = int(len(lines) * 0.9)
        for line in lines[:idx]:
            train.write(line)
        for line in lines[idx:]:
            dev.write(line)


def combine_files(one, two, out):
    if os.path.exists(out):
        os.remove(out)
    with open(one, encoding="utf-8") as one, open(two, encoding="utf-8") as two, open(
        out, "a", encoding="utf-8"
    ) as out:
        for line in one:
            out.write(line)
        for line in two:
            out.write(line)


def bmes_tag(input_file, output_file):
    with open(input_file, encoding="utf-8") as input_data, open(
        output_file, "w", encoding="utf-8"
    ) as output_data:
        for line in input_data:
            word_list = line.strip().split()
            for word in word_list:
                if len(word) == 1 or (
                    len(word) > 2 and word[0] == "<" and word[-1] == ">"
                ):
                    output_data.write(word + "\tS\n")
                else:
                    output_data.write(word[0] + "\tB\n")
                    for w in word[1 : len(word) - 1]:
                        output_data.write(w + "\tM\n")
                    output_data.write(word[len(word) - 1] + "\tE\n")
            output_data.write("\n")


def make_bmes(dataset="pku"):
    path = data_path + "/" + dataset + "/"
    make_sure_path_exists(path + "bmes")
    bmes_tag(path + "raw/train.txt", path + "bmes/train.txt")
    bmes_tag(path + "raw/train-all.txt", path + "bmes/train-all.txt")
    bmes_tag(path + "raw/dev.txt", path + "bmes/dev.txt")
    bmes_tag(path + "raw/test.txt", path + "bmes/test.txt")


def convert_sighan2005_dataset(dataset):
    global sighan05_root
    root = os.path.join(data_path, dataset)
    make_sure_path_exists(root)
    make_sure_path_exists(root + "/raw")
    file_path = "{}/{}_training.utf8".format(sighan05_root, dataset)
    convert_file(
        file_path, "{}/raw/train-all.txt".format(root), is_traditional(dataset), True
    )
    if dataset == "as":
        file_path = "{}/{}_testing_gold.utf8".format(sighan05_root, dataset)
    else:
        file_path = "{}/{}_test_gold.utf8".format(sighan05_root, dataset)
    convert_file(
        file_path, "{}/raw/test.txt".format(root), is_traditional(dataset), False
    )
    split_train_dev(dataset)


def convert_sighan2008_dataset(dataset, utf=16):
    global sighan08_root
    root = os.path.join(data_path, dataset)
    make_sure_path_exists(root)
    make_sure_path_exists(root + "/raw")
    convert_file(
        "{}/{}_train_utf{}.seg".format(sighan08_root, dataset, utf),
        "{}/raw/train-all.txt".format(root),
        is_traditional(dataset),
        True,
        "utf-{}".format(utf),
    )
    convert_file(
        "{}/{}_seg_truth&resource/{}_truth_utf{}.seg".format(
            sighan08_root, dataset, dataset, utf
        ),
        "{}/raw/test.txt".format(root),
        is_traditional(dataset),
        False,
        "utf-{}".format(utf),
    )
    split_train_dev(dataset)


def extract_conll(src, out):
    words = []
    with open(src, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as out:
        for line in src:
            line = line.strip()
            if len(line) == 0:
                out.write(" ".join(words) + "\n")
                words = []
                continue
            cells = line.split()
            words.append(cells[1])


def make_joint_corpus(datasets, joint):
    parts = ["dev", "test", "train", "train-all"]
    for part in parts:
        old_file = "{}/{}/raw/{}.txt".format(data_path, joint, part)
        if os.path.exists(old_file):
            os.remove(old_file)
        elif not os.path.exists(os.path.dirname(old_file)):
            os.makedirs(os.path.dirname(old_file))
        for name in datasets:
            append_tags(
                os.path.join(data_path, name, "raw"),
                os.path.dirname(old_file),
                name,
                part,
                encode="utf-8",
            )


def convert_all_sighan2005(datasets):
    for dataset in datasets:
        print(("Converting sighan bakeoff 2005 corpus: {}".format(dataset)))
        convert_sighan2005_dataset(dataset)
        make_bmes(dataset)


def convert_all_sighan2008(datasets):
    for dataset in datasets:
        print(("Converting sighan bakeoff 2008 corpus: {}".format(dataset)))
        convert_sighan2008_dataset(dataset, 16)
        make_bmes(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--sighan05", required=True, type=str, help="path to sighan2005 dataset")
    parser.add_argument("--sighan08", required=True, type=str, help="path to sighan2008 dataset")
    parser.add_argument("--data_path", required=True, type=str, help="path to save dataset")
    # fmt: on

    args, _ = parser.parse_known_args()
    sighan05_root = args.sighan05
    sighan08_root = args.sighan08
    data_path = args.data_path

    print("Converting sighan2005 Simplified Chinese corpus")
    datasets = "pku", "msr", "as", "cityu"
    convert_all_sighan2005(datasets)

    print("Combining sighan2005 corpus to one joint Simplified Chinese corpus")
    datasets = "pku", "msr", "as", "cityu"
    make_joint_corpus(datasets, "joint-sighan2005")
    make_bmes("joint-sighan2005")

    # For researchers who have access to sighan2008 corpus, use official corpora please.
    print("Converting sighan2008 Simplified Chinese corpus")
    datasets = "ctb", "ckip", "cityu", "ncc", "sxu"
    convert_all_sighan2008(datasets)
    print("Combining those 8 sighan corpora to one joint corpus")
    datasets = "pku", "msr", "as", "ctb", "ckip", "cityu", "ncc", "sxu"
    make_joint_corpus(datasets, "joint-sighan2008")
    make_bmes("joint-sighan2008")


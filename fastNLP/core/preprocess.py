import _pickle
import os

import numpy as np

DEFAULT_PADDING_LABEL = '<pad>'  # dict index = 0
DEFAULT_UNKNOWN_LABEL = '<unk>'  # dict index = 1
DEFAULT_RESERVED_LABEL = ['<reserved-2>',
                          '<reserved-3>',
                          '<reserved-4>']  # dict index = 2~4

DEFAULT_WORD_TO_INDEX = {DEFAULT_PADDING_LABEL: 0, DEFAULT_UNKNOWN_LABEL: 1,
                         DEFAULT_RESERVED_LABEL[0]: 2, DEFAULT_RESERVED_LABEL[1]: 3,
                         DEFAULT_RESERVED_LABEL[2]: 4}


# the first vocab in dict with the index = 5

def save_pickle(obj, pickle_path, file_name):
    with open(os.path.join(pickle_path, file_name), "wb") as f:
        _pickle.dump(obj, f)
    print("{} saved in {}".format(file_name, pickle_path))


def load_pickle(pickle_path, file_name):
    with open(os.path.join(pickle_path, file_name), "rb") as f:
        obj = _pickle.load(f)
    print("{} loaded from {}".format(file_name, pickle_path))
    return obj


def pickle_exist(pickle_path, pickle_name):
    """
    :param pickle_path: the directory of target pickle file
    :param pickle_name: the filename of target pickle file
    :return: True if file exists else False
    """
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    file_name = os.path.join(pickle_path, pickle_name)
    if os.path.exists(file_name):
        return True
    else:
        return False


class BasePreprocess(object):
    def __init__(self):
        self.word2index = None
        self.label2index = None

    @property
    def vocab_size(self):
        return len(self.word2index)

    @property
    def num_classes(self):
        return len(self.label2index)

    def run(self, train_dev_data, test_data=None, pickle_path="./", train_dev_split=0, cross_val=False, n_fold=10):
        """Main preprocessing pipeline.

        :param train_dev_data: three-level list, with either single label or multiple labels in a sample.
        :param test_data: three-level list, with either single label or multiple labels in a sample. (optional)
        :param pickle_path: str, the path to save the pickle files.
        :param train_dev_split: float, between [0, 1]. The ratio of training data used as validation set.
        :param cross_val: bool, whether to do cross validation.
        :param n_fold: int, the number of folds of cross validation. Only useful when cross_val is True.
        :return results: a tuple of datasets after preprocessing.
        """
        if pickle_exist(pickle_path, "word2id.pkl") and pickle_exist(pickle_path, "class2id.pkl"):
            self.word2index = load_pickle(pickle_path, "word2id.pkl")
            self.label2index = load_pickle(pickle_path, "class2id.pkl")
        else:
            self.word2index, self.label2index = self.build_dict(train_dev_data)
            save_pickle(self.word2index, pickle_path, "word2id.pkl")
            save_pickle(self.label2index, pickle_path, "class2id.pkl")

        if not pickle_exist(pickle_path, "id2word.pkl"):
            index2word = self.build_reverse_dict(self.word2index)
            save_pickle(index2word, pickle_path, "id2word.pkl")

        if not pickle_exist(pickle_path, "id2class.pkl"):
            index2label = self.build_reverse_dict(self.label2index)
            save_pickle(index2label, pickle_path, "id2class.pkl")

        data_train = []
        data_dev = []
        if not cross_val:
            if not pickle_exist(pickle_path, "data_train.pkl"):
                data_train.extend(self.to_index(train_dev_data))
                if train_dev_split > 0 and not pickle_exist(pickle_path, "data_dev.pkl"):
                    split = int(len(data_train) * train_dev_split)
                    data_dev = data_train[: split]
                    data_train = data_train[split:]
                    save_pickle(data_dev, pickle_path, "data_dev.pkl")
                    print("{} of the training data is split for validation. ".format(train_dev_split))
                save_pickle(data_train, pickle_path, "data_train.pkl")
            else:
                data_train = load_pickle(pickle_path, "data_train.pkl")
        else:
            # cross_val is True
            if not pickle_exist(pickle_path, "data_train_0.pkl"):
                # cross validation
                data_idx = self.to_index(train_dev_data)
                data_cv = self.cv_split(data_idx, n_fold)
                for i, (data_train_cv, data_dev_cv) in enumerate(data_cv):
                    save_pickle(
                        data_train_cv, pickle_path,
                        "data_train_{}.pkl".format(i))
                    save_pickle(
                        data_dev_cv, pickle_path,
                        "data_dev_{}.pkl".format(i))
                    data_train.append(data_train_cv)
                    data_dev.append(data_dev_cv)
                print("{}-fold cross validation.".format(n_fold))
            else:
                for i in range(n_fold):
                    data_train_cv = load_pickle(pickle_path, "data_train_{}.pkl".format(i))
                    data_dev_cv = load_pickle(pickle_path, "data_dev_{}.pkl".format(i))
                    data_train.append(data_train_cv)
                    data_dev.append(data_dev_cv)

        # prepare test data if provided
        data_test = []
        if test_data is not None:
            if not pickle_exist(pickle_path, "data_test.pkl"):
                data_test = self.to_index(test_data)
                save_pickle(data_test, pickle_path, "data_test.pkl")

        # return preprocessed results
        results = [data_train]
        if cross_val or train_dev_split > 0:
            results.append(data_dev)
        if test_data:
            results.append(data_test)
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def build_dict(self, data):
        raise NotImplementedError

    def to_index(self, data):
        raise NotImplementedError

    def build_reverse_dict(self, word_dict):
        id2word = {word_dict[w]: w for w in word_dict}
        return id2word

    def data_split(self, data, train_dev_split):
        """Split data into train and dev set."""
        split = int(len(data) * train_dev_split)
        data_dev = data[: split]
        data_train = data[split:]
        return data_train, data_dev

    def cv_split(self, data, n_fold):
        """Split data for cross validation."""
        data_copy = data.copy()
        np.random.shuffle(data_copy)
        fold_size = round(len(data_copy) / n_fold)

        data_cv = []
        for i in range(n_fold - 1):
            start = i * fold_size
            end = (i + 1) * fold_size
            data_dev = data_copy[start:end]
            data_train = data_copy[:start] + data_copy[end:]
            data_cv.append((data_train, data_dev))
        start = (n_fold - 1) * fold_size
        data_dev = data_copy[start:]
        data_train = data_copy[:start]
        data_cv.append((data_train, data_dev))

        return data_cv


class SeqLabelPreprocess(BasePreprocess):
    """Preprocess pipeline, including building mapping from words to index, from index to words,
        from labels/classes to index, from index to labels/classes.
        data of three-level list which have multiple labels in each sample.
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]
    """

    def __init__(self):
        super(SeqLabelPreprocess, self).__init__()

    def build_dict(self, data):
        """
        Add new words with indices into self.word_dict, new labels with indices into self.label_dict.
        :param data: three-level list
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]
        :return word2index: dict of {str, int}
                label2index: dict of {str, int}
        """
        # In seq labeling, both word seq and label seq need to be padded to the same length in a mini-batch.
        label2index = DEFAULT_WORD_TO_INDEX.copy()
        word2index = DEFAULT_WORD_TO_INDEX.copy()
        for example in data:
            for word, label in zip(example[0], example[1]):
                if word not in word2index:
                    word2index[word] = len(word2index)
                if label not in label2index:
                    label2index[label] = len(label2index)
        return word2index, label2index

    def to_index(self, data):
        """
        Convert word strings and label strings into indices.
        :param data: three-level list
            [
                [ [word_11, word_12, ...], [label_1, label_1, ...] ],
                [ [word_21, word_22, ...], [label_2, label_1, ...] ],
                ...
            ]
        :return data_index: the same shape as data, but each string is replaced by its corresponding index
        """
        data_index = []
        for example in data:
            word_list = []
            label_list = []
            for word, label in zip(example[0], example[1]):
                word_list.append(self.word2index.get(word, DEFAULT_WORD_TO_INDEX[DEFAULT_UNKNOWN_LABEL]))
                label_list.append(self.label2index.get(label, DEFAULT_WORD_TO_INDEX[DEFAULT_UNKNOWN_LABEL]))
            data_index.append([word_list, label_list])
        return data_index


class ClassPreprocess(BasePreprocess):
    """ Preprocess pipeline for classification datasets.
        Preprocess pipeline, including building mapping from words to index, from index to words,
        from labels/classes to index, from index to labels/classes.
        design for data of three-level list which has a single label in each sample.
            [
                [ [word_11, word_12, ...], label_1 ],
                [ [word_21, word_22, ...], label_2 ],
                ...
            ]
    """

    def __init__(self):
        super(ClassPreprocess, self).__init__()

    def build_dict(self, data):
        """Build vocabulary."""

        # build vocabulary from scratch if nothing exists
        word2index = DEFAULT_WORD_TO_INDEX.copy()
        label2index = DEFAULT_WORD_TO_INDEX.copy()

        # collect every word and label
        for sent, label in data:
            if len(sent) <= 1:
                continue

            if label not in label2index:
                label2index[label] = len(label2index)

            for word in sent:
                if word not in word2index:
                    word2index[word] = len(word2index)
        return word2index, label2index

    def to_index(self, data):
        """
        Convert word strings and label strings into indices.
        :param data: three-level list
            [
                [ [word_11, word_12, ...], label_1 ],
                [ [word_21, word_22, ...], label_2 ],
                ...
            ]
        :return data_index: the same shape as data, but each string is replaced by its corresponding index
        """
        data_index = []
        for example in data:
            word_list = []
            # example[0] is the word list, example[1] is the single label
            for word in example[0]:
                word_list.append(self.word2index.get(word, DEFAULT_WORD_TO_INDEX[DEFAULT_UNKNOWN_LABEL]))
            label_index = self.label2index.get(example[1], DEFAULT_WORD_TO_INDEX[DEFAULT_UNKNOWN_LABEL])
            data_index.append([word_list, label_index])
        return data_index


def infer_preprocess(pickle_path, data):
    """
        Preprocess over inference data.
        Transform three-level list of strings into that of index.
        [
            [word_11, word_12, ...],
            [word_21, word_22, ...],
            ...
        ]
    """
    word2index = load_pickle(pickle_path, "word2id.pkl")
    data_index = []
    for example in data:
        data_index.append([word2index.get(w, DEFAULT_UNKNOWN_LABEL) for w in example])
    return data_index

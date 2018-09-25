import _pickle
import os

import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary


# the first vocab in dict with the index = 5

def save_pickle(obj, pickle_path, file_name):
    """Save an object into a pickle file.

    :param obj: an object
    :param pickle_path: str, the directory where the pickle file is to be saved
    :param file_name: str, the name of the pickle file. In general, it should be ended by "pkl".
    """
    with open(os.path.join(pickle_path, file_name), "wb") as f:
        _pickle.dump(obj, f)
    print("{} saved in {}".format(file_name, pickle_path))


def load_pickle(pickle_path, file_name):
    """Load an object from a given pickle file.

    :param pickle_path: str, the directory where the pickle file is.
    :param file_name: str, the name of the pickle file.
    :return obj: an object stored in the pickle
    """
    with open(os.path.join(pickle_path, file_name), "rb") as f:
        obj = _pickle.load(f)
    print("{} loaded from {}".format(file_name, pickle_path))
    return obj


def pickle_exist(pickle_path, pickle_name):
    """Check if a given pickle file exists in the directory.

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


class Preprocessor(object):
    """Preprocessors are responsible for converting data of strings into data of indices.
    During the pre-processing, the following pickle files will be built:

        - "word2id.pkl", a Vocabulary object, mapping words to indices.
        - "class2id.pkl", a Vocabulary object, mapping labels to indices.
        - "data_train.pkl", a DataSet object for training
        - "data_dev.pkl", a DataSet object for validation, if train_dev_split > 0.
        - "data_test.pkl", a DataSet object for testing, if test_data is not None.

    These four pickle files are expected to be saved in the given pickle directory once they are constructed.
    Preprocessors will check if those files are already in the directory and will reuse them in future calls.
    """

    def __init__(self, label_is_seq=False):
        """

        :param label_is_seq: bool, whether label is a sequence. If True, label vocabulary will preserve
                several special tokens for sequence processing.
        """
        self.data_vocab = Vocabulary()
        self.label_vocab = Vocabulary(need_default=label_is_seq)

    @property
    def vocab_size(self):
        return len(self.data_vocab)

    @property
    def num_classes(self):
        return len(self.label_vocab)

    def run(self, train_dev_data, test_data=None, pickle_path="./", train_dev_split=0, cross_val=False, n_fold=10):
        """Main pre-processing pipeline.

        :param train_dev_data: three-level list, with either single label or multiple labels in a sample.
        :param test_data: three-level list, with either single label or multiple labels in a sample. (optional)
        :param pickle_path: str, the path to save the pickle files.
        :param train_dev_split: float, between [0, 1]. The ratio of training data used as validation set.
        :param cross_val: bool, whether to do cross validation.
        :param n_fold: int, the number of folds of cross validation. Only useful when cross_val is True.
        :return results: multiple datasets after pre-processing. If test_data is provided, return one more dataset.
                If train_dev_split > 0, return one more dataset - the dev set. If cross_val is True, each dataset
                is a list of DataSet objects; Otherwise, each dataset is a DataSet object.
        """

        if pickle_exist(pickle_path, "word2id.pkl") and pickle_exist(pickle_path, "class2id.pkl"):
            self.data_vocab = load_pickle(pickle_path, "word2id.pkl")
            self.label_vocab = load_pickle(pickle_path, "class2id.pkl")
        else:
            self.data_vocab, self.label_vocab = self.build_dict(train_dev_data)
            save_pickle(self.data_vocab, pickle_path, "word2id.pkl")
            save_pickle(self.label_vocab, pickle_path, "class2id.pkl")

        train_set = []
        dev_set = []
        if not cross_val:
            if not pickle_exist(pickle_path, "data_train.pkl"):
                if train_dev_split > 0 and not pickle_exist(pickle_path, "data_dev.pkl"):
                    split = int(len(train_dev_data) * train_dev_split)
                    data_dev = train_dev_data[: split]
                    data_train = train_dev_data[split:]
                    train_set = self.convert_to_dataset(data_train, self.data_vocab, self.label_vocab)
                    dev_set = self.convert_to_dataset(data_dev, self.data_vocab, self.label_vocab)

                    save_pickle(dev_set, pickle_path, "data_dev.pkl")
                    print("{} of the training data is split for validation. ".format(train_dev_split))
                else:
                    train_set = self.convert_to_dataset(train_dev_data, self.data_vocab, self.label_vocab)
                save_pickle(train_set, pickle_path, "data_train.pkl")
            else:
                train_set = load_pickle(pickle_path, "data_train.pkl")
                if pickle_exist(pickle_path, "data_dev.pkl"):
                    dev_set = load_pickle(pickle_path, "data_dev.pkl")
        else:
            # cross_val is True
            if not pickle_exist(pickle_path, "data_train_0.pkl"):
                # cross validation
                data_cv = self.cv_split(train_dev_data, n_fold)
                for i, (data_train_cv, data_dev_cv) in enumerate(data_cv):
                    data_train_cv = self.convert_to_dataset(data_train_cv, self.data_vocab, self.label_vocab)
                    data_dev_cv = self.convert_to_dataset(data_dev_cv, self.data_vocab, self.label_vocab)
                    save_pickle(
                        data_train_cv, pickle_path,
                        "data_train_{}.pkl".format(i))
                    save_pickle(
                        data_dev_cv, pickle_path,
                        "data_dev_{}.pkl".format(i))
                    train_set.append(data_train_cv)
                    dev_set.append(data_dev_cv)
                print("{}-fold cross validation.".format(n_fold))
            else:
                for i in range(n_fold):
                    data_train_cv = load_pickle(pickle_path, "data_train_{}.pkl".format(i))
                    data_dev_cv = load_pickle(pickle_path, "data_dev_{}.pkl".format(i))
                    train_set.append(data_train_cv)
                    dev_set.append(data_dev_cv)

        # prepare test data if provided
        test_set = []
        if test_data is not None:
            if not pickle_exist(pickle_path, "data_test.pkl"):
                test_set = self.convert_to_dataset(test_data, self.data_vocab, self.label_vocab)
                save_pickle(test_set, pickle_path, "data_test.pkl")

        # return preprocessed results
        results = [train_set]
        if cross_val or train_dev_split > 0:
            results.append(dev_set)
        if test_data:
            results.append(test_set)
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def build_dict(self, data):
        for example in data:
            word, label = example
            self.data_vocab.update(word)
            self.label_vocab.update(label)
        return self.data_vocab, self.label_vocab

    def build_reverse_dict(self):
        self.data_vocab.build_reverse_vocab()
        self.label_vocab.build_reverse_vocab()

    def data_split(self, data, train_dev_split):
        """Split data into train and dev set."""
        split = int(len(data) * train_dev_split)
        data_dev = data[: split]
        data_train = data[split:]
        return data_train, data_dev

    def cv_split(self, data, n_fold):
        """Split data for cross validation.

        :param data: list of string
        :param n_fold: int
        :return data_cv:

            ::
            [
                (data_train, data_dev),  # 1st fold
                (data_train, data_dev),  # 2nd fold
                ...
            ]

        """
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

    def convert_to_dataset(self, data, vocab, label_vocab):
        """Convert list of indices into a DataSet object.

        :param data: list. Entries are strings.
        :param vocab: a dict, mapping string (token) to index (int).
        :param label_vocab: a dict, mapping string (label) to index (int).
        :return data_set: a DataSet object
        """
        use_word_seq = False
        use_label_seq = False
        use_label_str = False

        # construct a DataSet object and fill it with Instances
        data_set = DataSet()
        for example in data:
            words, label = example[0], example[1]
            instance = Instance()

            if isinstance(words, list):
                x = TextField(words, is_target=False)
                instance.add_field("word_seq", x)
                use_word_seq = True
            else:
                raise NotImplementedError("words is a {}".format(type(words)))

            if isinstance(label, list):
                y = TextField(label, is_target=True)
                instance.add_field("label_seq", y)
                use_label_seq = True
            elif isinstance(label, str):
                y = LabelField(label, is_target=True)
                instance.add_field("label", y)
                use_label_str = True
            else:
                raise NotImplementedError("label is a {}".format(type(label)))
            data_set.append(instance)

        # convert strings to indices
        if use_word_seq:
            data_set.index_field("word_seq", vocab)
        if use_label_seq:
            data_set.index_field("label_seq", label_vocab)
        if use_label_str:
            data_set.index_field("label", label_vocab)

        return data_set


class SeqLabelPreprocess(Preprocessor):
    def __init__(self):
        print("[FastNLP warning] SeqLabelPreprocess is about to deprecate. Please use Preprocess directly.")
        super(SeqLabelPreprocess, self).__init__()


class ClassPreprocess(Preprocessor):
    def __init__(self):
        print("[FastNLP warning] ClassPreprocess is about to deprecate. Please use Preprocess directly.")
        super(ClassPreprocess, self).__init__()


if __name__ == "__main__":
    p = Preprocessor()
    train_dev_data = [[["I", "am", "a", "good", "student", "."], "0"],
                      [["You", "are", "pretty", "."], "1"]
                      ]
    training_set = p.run(train_dev_data)
    print(training_set)

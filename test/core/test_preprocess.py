import os
import unittest

from fastNLP.core.dataset import DataSet
from fastNLP.core.preprocess import SeqLabelPreprocess

data = [
    [['Tom', 'and', 'Jerry', '.'], ['n', '&', 'n', '.']],
    [['Hello', 'world', '!'], ['a', 'n', '.']],
    [['Tom', 'and', 'Jerry', '.'], ['n', '&', 'n', '.']],
    [['Hello', 'world', '!'], ['a', 'n', '.']],
    [['Tom', 'and', 'Jerry', '.'], ['n', '&', 'n', '.']],
    [['Hello', 'world', '!'], ['a', 'n', '.']],
    [['Tom', 'and', 'Jerry', '.'], ['n', '&', 'n', '.']],
    [['Hello', 'world', '!'], ['a', 'n', '.']],
    [['Tom', 'and', 'Jerry', '.'], ['n', '&', 'n', '.']],
    [['Hello', 'world', '!'], ['a', 'n', '.']],
]


class TestCase1(unittest.TestCase):
    def test(self):
        if os.path.exists("./save"):
            for root, dirs, files in os.walk("./save", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        result = SeqLabelPreprocess().run(train_dev_data=data, train_dev_split=0.4,
                                          pickle_path="./save")
        self.assertEqual(len(result), 2)
        self.assertEqual(type(result[0]), DataSet)
        self.assertEqual(type(result[1]), DataSet)

        os.system("rm -rf save")
        print("pickle path deleted")


class TestCase2(unittest.TestCase):
    def test(self):
        if os.path.exists("./save"):
            for root, dirs, files in os.walk("./save", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        result = SeqLabelPreprocess().run(test_data=data, train_dev_data=data,
                                          pickle_path="./save", train_dev_split=0.4,
                                          cross_val=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(type(result[0]), DataSet)
        self.assertEqual(type(result[1]), DataSet)
        self.assertEqual(type(result[2]), DataSet)

        os.system("rm -rf save")
        print("pickle path deleted")


class TestCase3(unittest.TestCase):
    def test(self):
        num_folds = 2
        result = SeqLabelPreprocess().run(test_data=None, train_dev_data=data,
                                          pickle_path="./save", train_dev_split=0.4,
                                          cross_val=True, n_fold=num_folds)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), num_folds)
        self.assertEqual(len(result[1]), num_folds)
        for data_set in result[0] + result[1]:
            self.assertEqual(type(data_set), DataSet)

        os.system("rm -rf save")
        print("pickle path deleted")

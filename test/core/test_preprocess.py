import os
import unittest

from fastNLP.core.preprocess import SeqLabelPreprocess


class TestSeqLabelPreprocess(unittest.TestCase):
    def test_case_1(self):
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

        if os.path.exists("./save"):
            for root, dirs, files in os.walk("./save", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        result = SeqLabelPreprocess().run(train_dev_data=data, train_dev_split=0.4,
                                          pickle_path="./save")
        result = SeqLabelPreprocess().run(train_dev_data=data, train_dev_split=0.4,
                                          pickle_path="./save")
        if os.path.exists("./save"):
            for root, dirs, files in os.walk("./save", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        result = SeqLabelPreprocess().run(test_data=data, train_dev_data=data,
                                                           pickle_path="./save", train_dev_split=0.4,
                                                           cross_val=True)
        result = SeqLabelPreprocess().run(test_data=data, train_dev_data=data,
                                          pickle_path="./save", train_dev_split=0.4,
                                          cross_val=True)

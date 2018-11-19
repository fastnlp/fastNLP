import unittest


class TestDataSet(unittest.TestCase):
    labeled_data_list = [
        [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
        [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
        [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
    ]
    unlabeled_data_list = [
        ["a", "b", "e", "d"],
        ["a", "b", "e", "d"],
        ["a", "b", "e", "d"]
    ]
    word_vocab = {"a": 0, "b": 1, "e": 2, "d": 3}
    label_vocab = {"1": 1, "2": 2, "3": 3, "4": 4}

    def test_case_1(self):
        # TODO:
        pass

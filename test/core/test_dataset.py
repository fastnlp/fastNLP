import unittest

from fastNLP.core.dataset import SeqLabelDataSet, TextClassifyDataSet
from fastNLP.core.dataset import create_dataset_from_lists


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
        data_set = create_dataset_from_lists(self.labeled_data_list, self.word_vocab, has_target=True,
                                             label_vocab=self.label_vocab)
        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.labeled_data_list[0][0]])

        self.assertTrue("label_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["label_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["label_seq"], "_index"))
        self.assertEqual(data_set[0].fields["label_seq"].text, self.labeled_data_list[0][1])
        self.assertEqual(data_set[0].fields["label_seq"]._index,
                         [self.label_vocab[c] for c in self.labeled_data_list[0][1]])

    def test_case_2(self):
        data_set = create_dataset_from_lists(self.unlabeled_data_list, self.word_vocab, has_target=False)

        self.assertEqual(len(data_set), len(self.unlabeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.unlabeled_data_list[0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.unlabeled_data_list[0]])


class TestDataSetConvertion(unittest.TestCase):
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
        def loader(path):
            labeled_data_list = [
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
            ]
            return labeled_data_list

        data_set = SeqLabelDataSet(load_func=loader)
        data_set.load("any_path")

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)

        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])

        self.assertTrue("truth" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["truth"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["truth"], "_index"))
        self.assertEqual(data_set[0].fields["truth"].text, self.labeled_data_list[0][1])

        self.assertTrue("word_seq_origin_len" in data_set[0].fields)

    def test_case_2(self):
        def loader(path):
            unlabeled_data_list = [
                ["a", "b", "e", "d"],
                ["a", "b", "e", "d"],
                ["a", "b", "e", "d"]
            ]
            return unlabeled_data_list

        data_set = SeqLabelDataSet(load_func=loader)
        data_set.load("any_path", vocabs={"word_vocab": self.word_vocab}, infer=True)

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.labeled_data_list[0][0]])

        self.assertTrue("word_seq_origin_len" in data_set[0].fields)

    def test_case_3(self):
        def loader(path):
            labeled_data_list = [
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
                [["a", "b", "e", "d"], ["1", "2", "3", "4"]],
            ]
            return labeled_data_list

        data_set = SeqLabelDataSet(load_func=loader)
        data_set.load("any_path", vocabs={"word_vocab": self.word_vocab, "label_vocab": self.label_vocab})

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.labeled_data_list[0][0]])

        self.assertTrue("truth" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["truth"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["truth"], "_index"))
        self.assertEqual(data_set[0].fields["truth"].text, self.labeled_data_list[0][1])
        self.assertEqual(data_set[0].fields["truth"]._index,
                         [self.label_vocab[c] for c in self.labeled_data_list[0][1]])

        self.assertTrue("word_seq_origin_len" in data_set[0].fields)


class TestDataSetConvertionHHH(unittest.TestCase):
    labeled_data_list = [
        [["a", "b", "e", "d"], "A"],
        [["a", "b", "e", "d"], "C"],
        [["a", "b", "e", "d"], "B"],
    ]
    unlabeled_data_list = [
        ["a", "b", "e", "d"],
        ["a", "b", "e", "d"],
        ["a", "b", "e", "d"]
    ]
    word_vocab = {"a": 0, "b": 1, "e": 2, "d": 3}
    label_vocab = {"A": 1, "B": 2, "C": 3}

    def test_case_1(self):
        def loader(path):
            labeled_data_list = [
                [["a", "b", "e", "d"], "A"],
                [["a", "b", "e", "d"], "C"],
                [["a", "b", "e", "d"], "B"],
            ]
            return labeled_data_list

        data_set = TextClassifyDataSet(load_func=loader)
        data_set.load("xxx")

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)

        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])

        self.assertTrue("label" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["label"], "label"))
        self.assertTrue(hasattr(data_set[0].fields["label"], "_index"))
        self.assertEqual(data_set[0].fields["label"].label, self.labeled_data_list[0][1])

    def test_case_2(self):
        def loader(path):
            labeled_data_list = [
                [["a", "b", "e", "d"], "A"],
                [["a", "b", "e", "d"], "C"],
                [["a", "b", "e", "d"], "B"],
            ]
            return labeled_data_list

        data_set = TextClassifyDataSet(load_func=loader)
        data_set.load("xxx", vocabs={"word_vocab": self.word_vocab, "label_vocab": self.label_vocab})

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)

        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.labeled_data_list[0][0]])

        self.assertTrue("label" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["label"], "label"))
        self.assertTrue(hasattr(data_set[0].fields["label"], "_index"))
        self.assertEqual(data_set[0].fields["label"].label, self.labeled_data_list[0][1])
        self.assertEqual(data_set[0].fields["label"]._index, self.label_vocab[self.labeled_data_list[0][1]])

    def test_case_3(self):
        def loader(path):
            unlabeled_data_list = [
                ["a", "b", "e", "d"],
                ["a", "b", "e", "d"],
                ["a", "b", "e", "d"]
            ]
            return unlabeled_data_list

        data_set = TextClassifyDataSet(load_func=loader)
        data_set.load("xxx", vocabs={"word_vocab": self.word_vocab}, infer=True)

        self.assertEqual(len(data_set), len(self.labeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)

        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.labeled_data_list[0][0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.labeled_data_list[0][0]])

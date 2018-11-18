import unittest

from fastNLP.io.dataset_loader import convert_seq2seq_dataset, convert_seq_dataset


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
        data_set = convert_seq2seq_dataset(self.labeled_data_list)
        data_set.index_field("word_seq", self.word_vocab)
        data_set.index_field("label_seq", self.label_vocab)
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
        data_set = convert_seq_dataset(self.unlabeled_data_list)
        data_set.index_field("word_seq", self.word_vocab)

        self.assertEqual(len(data_set), len(self.unlabeled_data_list))
        self.assertTrue(len(data_set) > 0)
        self.assertTrue(hasattr(data_set[0], "fields"))
        self.assertTrue("word_seq" in data_set[0].fields)
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "text"))
        self.assertTrue(hasattr(data_set[0].fields["word_seq"], "_index"))
        self.assertEqual(data_set[0].fields["word_seq"].text, self.unlabeled_data_list[0])
        self.assertEqual(data_set[0].fields["word_seq"]._index,
                         [self.word_vocab[c] for c in self.unlabeled_data_list[0]])


import random
import unittest

from fastNLP import Vocabulary
from fastNLP.api.processor import FullSpaceToHalfSpaceProcessor, PreAppendProcessor, SliceProcessor, Num2TagProcessor, \
    IndexerProcessor, VocabProcessor, SeqLenProcessor
from fastNLP.core.dataset import DataSet


class TestProcessor(unittest.TestCase):
    def test_FullSpaceToHalfSpaceProcessor(self):
        ds = DataSet({"word": ["０0, u１, u), (u２, u2"]})
        proc = FullSpaceToHalfSpaceProcessor("word")
        ds = proc(ds)
        self.assertEqual(ds.field_arrays["word"].content, ["00, u1, u), (u2, u2"])

    def test_PreAppendProcessor(self):
        ds = DataSet({"word": [["1234", "3456"], ["8789", "3464"]]})
        proc = PreAppendProcessor(data="abc", field_name="word")
        ds = proc(ds)
        self.assertEqual(ds.field_arrays["word"].content, [["abc", "1234", "3456"], ["abc", "8789", "3464"]])

    def test_SliceProcessor(self):
        ds = DataSet({"xx": [[random.randint(0, 10) for _ in range(30)]] * 40})
        proc = SliceProcessor(10, 20, 2, "xx", new_added_field_name="yy")
        ds = proc(ds)
        self.assertEqual(len(ds.field_arrays["yy"].content[0]), 5)

    def test_Num2TagProcessor(self):
        ds = DataSet({"num": [["99.9982", "2134.0"], ["0.002", "234"]]})
        proc = Num2TagProcessor("<num>", "num")
        ds = proc(ds)
        for data in ds.field_arrays["num"].content:
            for d in data:
                self.assertEqual(d, "<num>")

    def test_VocabProcessor_and_IndexerProcessor(self):
        ds = DataSet({"xx": [[str(random.randint(0, 10)) for _ in range(30)]] * 40})
        vocab_proc = VocabProcessor("xx")
        vocab_proc(ds)
        vocab = vocab_proc.vocab
        self.assertTrue(isinstance(vocab, Vocabulary))
        self.assertTrue(len(vocab) > 5)

        proc = IndexerProcessor(vocab, "xx", "yy")
        ds = proc(ds)
        for data in ds.field_arrays["yy"].content[0]:
            self.assertTrue(isinstance(data, int))

    def test_SeqLenProcessor(self):
        ds = DataSet({"xx": [[str(random.randint(0, 10)) for _ in range(30)]] * 10})
        proc = SeqLenProcessor("xx", "len")
        ds = proc(ds)
        for data in ds.field_arrays["len"].content:
            self.assertEqual(data, 30)

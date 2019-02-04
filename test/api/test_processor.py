import random
import unittest

import numpy as np

from fastNLP import Vocabulary, Instance
from fastNLP.api.processor import FullSpaceToHalfSpaceProcessor, PreAppendProcessor, SliceProcessor, Num2TagProcessor, \
    IndexerProcessor, VocabProcessor, SeqLenProcessor, ModelProcessor, Index2WordProcessor, SetTargetProcessor, \
    SetInputProcessor, VocabIndexerProcessor
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

    def test_ModelProcessor(self):
        from fastNLP.models.cnn_text_classification import CNNText
        model = CNNText(100, 100, 5)
        ins_list = []
        for _ in range(64):
            seq_len = np.random.randint(5, 30)
            ins_list.append(Instance(word_seq=[np.random.randint(0, 100) for _ in range(seq_len)], seq_lens=seq_len))
        data_set = DataSet(ins_list)
        data_set.set_input("word_seq", "seq_lens")
        proc = ModelProcessor(model)
        data_set = proc(data_set)
        self.assertTrue("pred" in data_set)

    def test_Index2WordProcessor(self):
        vocab = Vocabulary()
        vocab.add_word_lst(["a", "b", "c", "d", "e"])
        proc = Index2WordProcessor(vocab, "tag_id", "tag")
        data_set = DataSet([Instance(tag_id=[np.random.randint(0, 7) for _ in range(32)])])
        data_set = proc(data_set)
        self.assertTrue("tag" in data_set)

    def test_SetTargetProcessor(self):
        proc = SetTargetProcessor("a", "b", "c")
        data_set = DataSet({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        data_set = proc(data_set)
        self.assertTrue(data_set["a"].is_target)
        self.assertTrue(data_set["b"].is_target)
        self.assertTrue(data_set["c"].is_target)

    def test_SetInputProcessor(self):
        proc = SetInputProcessor("a", "b", "c")
        data_set = DataSet({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        data_set = proc(data_set)
        self.assertTrue(data_set["a"].is_input)
        self.assertTrue(data_set["b"].is_input)
        self.assertTrue(data_set["c"].is_input)

    def test_VocabIndexerProcessor(self):
        proc = VocabIndexerProcessor("word_seq", "word_ids")
        data_set = DataSet([Instance(word_seq=["a", "b", "c", "d", "e"])])
        data_set = proc(data_set)
        self.assertTrue("word_ids" in data_set)

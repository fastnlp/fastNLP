import unittest
from collections import Counter

from fastNLP.core.vocabulary import Vocabulary

text = ["FastNLP", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
        "works", "well", "in", "most", "cases", "scales", "well"]
counter = Counter(text)


class TestAdd(unittest.TestCase):
    def test_add(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        for word in text:
            vocab.add(word)
        self.assertEqual(vocab.word_count, counter)

    def test_add_word(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        for word in text:
            vocab.add_word(word)
        self.assertEqual(vocab.word_count, counter)

    def test_add_word_lst(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        vocab.add_word_lst(text)
        self.assertEqual(vocab.word_count, counter)

    def test_update(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        vocab.update(text)
        self.assertEqual(vocab.word_count, counter)


class TestIndexing(unittest.TestCase):
    def test_len(self):
        vocab = Vocabulary(max_size=None, min_freq=None, unknown=None, padding=None)
        vocab.update(text)
        self.assertEqual(len(vocab), len(counter))

    def test_contains(self):
        vocab = Vocabulary(max_size=None, min_freq=None, unknown=None, padding=None)
        vocab.update(text)
        self.assertTrue(text[-1] in vocab)
        self.assertFalse("~!@#" in vocab)
        self.assertEqual(text[-1] in vocab, vocab.has_word(text[-1]))
        self.assertEqual("~!@#" in vocab, vocab.has_word("~!@#"))

    def test_index(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        vocab.update(text)
        res = [vocab[w] for w in set(text)]
        self.assertEqual(len(res), len(set(res)))

        res = [vocab.to_index(w) for w in set(text)]
        self.assertEqual(len(res), len(set(res)))

    def test_to_word(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        vocab.update(text)
        self.assertEqual(text, [vocab.to_word(idx) for idx in [vocab[w] for w in text]])


class TestOther(unittest.TestCase):
    def test_additional_update(self):
        vocab = Vocabulary(max_size=None, min_freq=None)
        vocab.update(text)

        _ = vocab["well"]
        self.assertEqual(vocab.rebuild, False)

        vocab.add("hahaha")
        self.assertEqual(vocab.rebuild, True)

        _ = vocab["hahaha"]
        self.assertEqual(vocab.rebuild, False)
        self.assertTrue("hahaha" in vocab)

    def test_warning(self):
        vocab = Vocabulary(max_size=len(set(text)), min_freq=None)
        vocab.update(text)
        self.assertEqual(vocab.rebuild, True)
        print(len(vocab))
        self.assertEqual(vocab.rebuild, False)

        vocab.update(["hahahha", "hhh", "vvvv", "ass", "asss", "jfweiong", "eqgfeg", "feqfw"])
        # this will print a warning
        self.assertEqual(vocab.rebuild, True)

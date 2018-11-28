import unittest
from collections import Counter

from fastNLP.core.vocabulary import Vocabulary

text = ["FastNLP", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
        "works", "well", "in", "most", "cases", "scales", "well"]
counter = Counter(text)


class TestAdd(unittest.TestCase):
    def test_add(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        for word in text:
            vocab.add(word)
        self.assertEqual(vocab.word_count, counter)

    def test_add_word(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        for word in text:
            vocab.add_word(word)
        self.assertEqual(vocab.word_count, counter)

    def test_add_word_lst(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        vocab.add_word_lst(text)
        self.assertEqual(vocab.word_count, counter)

    def test_update(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        vocab.update(text)
        self.assertEqual(vocab.word_count, counter)


class TestIndexing(unittest.TestCase):
    def test_len(self):
        vocab = Vocabulary(need_default=False, max_size=None, min_freq=None)
        vocab.update(text)
        self.assertEqual(len(vocab), len(counter))

    def test_contains(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        vocab.update(text)
        self.assertTrue(text[-1] in vocab)
        self.assertFalse("~!@#" in vocab)
        self.assertEqual(text[-1] in vocab, vocab.has_word(text[-1]))
        self.assertEqual("~!@#" in vocab, vocab.has_word("~!@#"))

    def test_index(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        vocab.update(text)
        res = [vocab[w] for w in set(text)]
        self.assertEqual(len(res), len(set(res)))

        res = [vocab.to_index(w) for w in set(text)]
        self.assertEqual(len(res), len(set(res)))

    def test_to_word(self):
        vocab = Vocabulary(need_default=True, max_size=None, min_freq=None)
        vocab.update(text)
        self.assertEqual(text, [vocab.to_word(idx) for idx in [vocab[w] for w in text]])

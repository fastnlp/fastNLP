import unittest
from collections import Counter

from fastNLP import Vocabulary
from fastNLP import DataSet
from fastNLP import Instance

text = ["FastNLP", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
        "works", "well", "in", "most", "cases", "scales", "well"]
counter = Counter(text)


class TestAdd(unittest.TestCase):
    def test_add(self):
        vocab = Vocabulary()
        for word in text:
            vocab.add(word)
        self.assertEqual(vocab.word_count, counter)
    
    def test_add_word(self):
        vocab = Vocabulary()
        for word in text:
            vocab.add_word(word)
        self.assertEqual(vocab.word_count, counter)
    
    def test_add_word_lst(self):
        vocab = Vocabulary()
        vocab.add_word_lst(text)
        self.assertEqual(vocab.word_count, counter)
    
    def test_update(self):
        vocab = Vocabulary()
        vocab.update(text)
        self.assertEqual(vocab.word_count, counter)
    
    def test_from_dataset(self):
        start_char = 65
        num_samples = 10
        
        # 0 dim
        dataset = DataSet()
        for i in range(num_samples):
            ins = Instance(char=chr(start_char + i))
            dataset.append(ins)
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='char')
        for i in range(num_samples):
            self.assertEqual(vocab.to_index(chr(start_char + i)), i + 2)
        vocab.index_dataset(dataset, field_name='char')
        
        # 1 dim
        dataset = DataSet()
        for i in range(num_samples):
            ins = Instance(char=[chr(start_char + i)] * 6)
            dataset.append(ins)
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='char')
        for i in range(num_samples):
            self.assertEqual(vocab.to_index(chr(start_char + i)), i + 2)
        vocab.index_dataset(dataset, field_name='char')
        
        # 2 dim
        dataset = DataSet()
        for i in range(num_samples):
            ins = Instance(char=[[chr(start_char + i) for _ in range(6)] for _ in range(6)])
            dataset.append(ins)
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='char')
        for i in range(num_samples):
            self.assertEqual(vocab.to_index(chr(start_char + i)), i + 2)
        vocab.index_dataset(dataset, field_name='char')


class TestIndexing(unittest.TestCase):
    def test_len(self):
        vocab = Vocabulary(unknown=None, padding=None)
        vocab.update(text)
        self.assertEqual(len(vocab), len(counter))
    
    def test_contains(self):
        vocab = Vocabulary(unknown=None)
        vocab.update(text)
        self.assertTrue(text[-1] in vocab)
        self.assertFalse("~!@#" in vocab)
        self.assertEqual(text[-1] in vocab, vocab.has_word(text[-1]))
        self.assertEqual("~!@#" in vocab, vocab.has_word("~!@#"))
    
    def test_index(self):
        vocab = Vocabulary()
        vocab.update(text)
        res = [vocab[w] for w in set(text)]
        self.assertEqual(len(res), len(set(res)))
        
        res = [vocab.to_index(w) for w in set(text)]
        self.assertEqual(len(res), len(set(res)))
    
    def test_to_word(self):
        vocab = Vocabulary()
        vocab.update(text)
        self.assertEqual(text, [vocab.to_word(idx) for idx in [vocab[w] for w in text]])
    
    def test_iteration(self):
        vocab = Vocabulary(padding=None, unknown=None)
        text = ["FastNLP", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
                "works", "well", "in", "most", "cases", "scales", "well"]
        vocab.update(text)
        text = set(text)
        for word, idx in vocab:
            self.assertTrue(word in text)
            self.assertTrue(idx < len(vocab))


class TestOther(unittest.TestCase):
    def test_additional_update(self):
        vocab = Vocabulary()
        vocab.update(text)
        
        _ = vocab["well"]
        self.assertEqual(vocab.rebuild, False)
        
        vocab.add("hahaha")
        self.assertEqual(vocab.rebuild, True)
        
        _ = vocab["hahaha"]
        self.assertEqual(vocab.rebuild, False)
        self.assertTrue("hahaha" in vocab)
    
    def test_warning(self):
        vocab = Vocabulary(max_size=len(set(text)))
        vocab.update(text)
        self.assertEqual(vocab.rebuild, True)
        print(len(vocab))
        self.assertEqual(vocab.rebuild, False)
        
        vocab.update(["hahahha", "hhh", "vvvv", "ass", "asss", "jfweiong", "eqgfeg", "feqfw"])
        # this will print a warning
        self.assertEqual(vocab.rebuild, True)

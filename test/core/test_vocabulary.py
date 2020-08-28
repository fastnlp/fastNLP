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

    def test_from_dataset_no_entry(self):
        # 测试能否正确将no_create_entry正确设置
        dataset = DataSet()
        start_char = 65
        num_samples = 10
        test_dataset = DataSet()
        for i in range(num_samples):
            char = [chr(start_char + i)] * 6
            ins = Instance(char=char)
            dataset.append(ins)
            ins = Instance(char=[c+c for c in char])
            test_dataset.append(ins)
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='char', no_create_entry_dataset=test_dataset)
        vocab.index_dataset(dataset, field_name='char')
        for i in range(num_samples):
            self.assertEqual(True, vocab._is_word_no_create_entry(chr(start_char + i)+chr(start_char + i)))

    def test_no_entry(self):
        # 先建立vocabulary，然后变化no_create_entry, 测试能否正确识别
        text = ["FastNLP", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
                "works", "well", "in", "most", "cases", "scales", "well"]
        vocab = Vocabulary()
        vocab.add_word_lst(text)

        self.assertFalse(vocab._is_word_no_create_entry('FastNLP'))
        vocab.add_word('FastNLP', no_create_entry=True)
        self.assertFalse(vocab._is_word_no_create_entry('FastNLP'))

        vocab.add_word('fastnlp', no_create_entry=True)
        self.assertTrue(vocab._is_word_no_create_entry('fastnlp'))
        vocab.add_word('fastnlp', no_create_entry=False)
        self.assertFalse(vocab._is_word_no_create_entry('fastnlp'))

        vocab.add_word_lst(['1']*10, no_create_entry=True)
        self.assertTrue(vocab._is_word_no_create_entry('1'))
        vocab.add_word('1')
        self.assertFalse(vocab._is_word_no_create_entry('1'))


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

    def test_rebuild(self):
        # 测试build之后新加入词，原来的词顺序不变
        vocab = Vocabulary()
        text = [str(idx) for idx in range(10)]
        vocab.update(text)
        for i in text:
            self.assertEqual(int(i)+2, vocab.to_index(i))
        indexes = []
        for word, index in vocab:
            indexes.append((word, index))
        vocab.add_word_lst([str(idx) for idx in range(10, 13)])
        for idx, pair in enumerate(indexes):
            self.assertEqual(pair[1], vocab.to_index(pair[0]))
        for i in range(13):
            self.assertEqual(int(i)+2, vocab.to_index(str(i)))

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

    def test_save_and_load(self):
        fp = 'vocab_save_test.txt'
        try:
            # check word2idx没变，no_create_entry正常
            words = list('abcdefaddfdkjfe')
            no_create_entry = list('12342331')
            unk = '[UNK]'
            vocab = Vocabulary(unknown=unk, max_size=500)

            vocab.add_word_lst(words)
            vocab.add_word_lst(no_create_entry, no_create_entry=True)
            vocab.save(fp)

            new_vocab = Vocabulary.load(fp)

            for word, index in vocab:
                self.assertEqual(new_vocab.to_index(word), index)
            for word in no_create_entry:
                self.assertTrue(new_vocab._is_word_no_create_entry(word))
            for word in words:
                self.assertFalse(new_vocab._is_word_no_create_entry(word))
            for idx in range(len(vocab)):
                self.assertEqual(vocab.to_word(idx), new_vocab.to_word(idx))
            self.assertEqual(vocab.unknown, new_vocab.unknown)

            # 测试vocab中包含None的padding和unk
            vocab= Vocabulary(padding=None, unknown=None)
            words = list('abcdefaddfdkjfe')
            no_create_entry = list('12342331')

            vocab.add_word_lst(words)
            vocab.add_word_lst(no_create_entry, no_create_entry=True)
            vocab.save(fp)

            new_vocab = Vocabulary.load(fp)

            for word, index in vocab:
                self.assertEqual(new_vocab.to_index(word), index)
            for word in no_create_entry:
                self.assertTrue(new_vocab._is_word_no_create_entry(word))
            for word in words:
                self.assertFalse(new_vocab._is_word_no_create_entry(word))
            for idx in range(len(vocab)):
                self.assertEqual(vocab.to_word(idx), new_vocab.to_word(idx))
            self.assertEqual(vocab.unknown, new_vocab.unknown)

        finally:
            import os
            if os.path.exists(fp):
                os.remove(fp)

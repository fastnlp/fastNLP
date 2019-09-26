
import unittest
import os

from fastNLP.io import DataBundle
from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, QNLIPipe, QuoraPipe, MNLIPipe, \
    CNXNLIPipe, BQCorpusPipe, LCQMCPipe
from fastNLP.io.pipe.matching import SNLIBertPipe, RTEBertPipe, QNLIBertPipe, QuoraBertPipe, MNLIBertPipe, \
    CNXNLIBertPipe, BQCorpusBertPipe, LCQMCBertPipe


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestMatchingPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [SNLIPipe, RTEPipe, QNLIPipe, MNLIPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestMatchingBertPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [SNLIBertPipe, RTEBertPipe, QNLIBertPipe, MNLIBertPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)


class TestRunMatchingPipe(unittest.TestCase):

    def test_load(self):
        data_set_dict = {
            'RTE': ('test/data_for_tests/io/RTE', RTEPipe, RTEBertPipe, (5, 5, 5), (449, 2), True),
            'SNLI': ('test/data_for_tests/io/SNLI', SNLIPipe, SNLIBertPipe, (5, 5, 5), (110, 3), False),
            'QNLI': ('test/data_for_tests/io/QNLI', QNLIPipe, QNLIBertPipe, (5, 5, 5), (372, 2), True),
            'MNLI': ('test/data_for_tests/io/MNLI', MNLIPipe, MNLIBertPipe, (5, 5, 5, 5, 6), (459, 3), True),
            'BQCorpus': ('test/data_for_tests/io/BQCorpus', BQCorpusPipe, BQCorpusBertPipe, (5, 5, 5), (32, 2), False),
            'XNLI': ('test/data_for_tests/io/XNLI', CNXNLIPipe, CNXNLIBertPipe, (6, 8, 6), (39, 3), False),
            'LCQMC': ('test/data_for_tests/io/LCQMC', LCQMCPipe, LCQMCBertPipe, (5, 6, 6), (36, 2), False),
        }
        for k, v in data_set_dict.items():
            path, pipe1, pipe2, data_set, vocab, warns = v
            if warns:
                with self.assertWarns(Warning):
                    data_bundle1 = pipe1(tokenizer='raw').process_from_file(path)
                    data_bundle2 = pipe2(tokenizer='raw').process_from_file(path)
            else:
                data_bundle1 = pipe1(tokenizer='raw').process_from_file(path)
                data_bundle2 = pipe2(tokenizer='raw').process_from_file(path)

            self.assertTrue(isinstance(data_bundle1, DataBundle))
            self.assertEqual(len(data_set), data_bundle1.num_dataset)
            print(k)
            print(data_bundle1)
            print(data_bundle2)
            for x, y in zip(data_set, data_bundle1.iter_datasets()):
                name, dataset = y
                self.assertEqual(x, len(dataset))
            self.assertEqual(len(data_set), data_bundle2.num_dataset)
            for x, y in zip(data_set, data_bundle2.iter_datasets()):
                name, dataset = y
                self.assertEqual(x, len(dataset))

            self.assertEqual(len(vocab), data_bundle1.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                self.assertEqual(x, len(vocabs))
            self.assertEqual(len(vocab), data_bundle2.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                self.assertEqual(x + 1 if name == 'words' else x, len(vocabs))

    def test_spacy(self):
        data_set_dict = {
            'Quora': ('test/data_for_tests/io/Quora', QuoraPipe, QuoraBertPipe, (2, 2, 2), (93, 2)),
            }
        for k, v in data_set_dict.items():
            path, pipe1, pipe2, data_set, vocab = v

            data_bundle1 = pipe1(tokenizer='spacy').process_from_file(path)
            data_bundle2 = pipe2(tokenizer='spacy').process_from_file(path)

            self.assertTrue(isinstance(data_bundle1, DataBundle))
            self.assertEqual(len(data_set), data_bundle1.num_dataset)
            print(k)
            print(data_bundle1)
            print(data_bundle2)
            for x, y in zip(data_set, data_bundle1.iter_datasets()):
                name, dataset = y
                self.assertEqual(x, len(dataset))
            self.assertEqual(len(data_set), data_bundle2.num_dataset)
            for x, y in zip(data_set, data_bundle2.iter_datasets()):
                name, dataset = y
                self.assertEqual(x, len(dataset))

            self.assertEqual(len(vocab), data_bundle1.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                self.assertEqual(x, len(vocabs))
            self.assertEqual(len(vocab), data_bundle2.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                self.assertEqual(x + 1 if name == 'words' else x, len(vocabs))


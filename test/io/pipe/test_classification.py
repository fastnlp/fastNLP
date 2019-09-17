import unittest
import os

from fastNLP.io import DataBundle
from fastNLP.io.pipe.classification import SSTPipe, SST2Pipe, IMDBPipe, YelpFullPipe, YelpPolarityPipe
from fastNLP.io.pipe.classification import ChnSentiCorpPipe


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestClassificationPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [YelpPolarityPipe, SST2Pipe, IMDBPipe, YelpFullPipe,  SSTPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)


class TestRunPipe(unittest.TestCase):
    def test_load(self):
        for pipe in [IMDBPipe]:
            data_bundle = pipe(tokenizer='raw').process_from_file('test/data_for_tests/io/imdb')
            print(data_bundle)


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestCNClassificationPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [ChnSentiCorpPipe]:
            with self.subTest(pipe=pipe):
                data_bundle = pipe(bigrams=True, trigrams=True).process_from_file()
                print(data_bundle)


class TestRunClassificationPipe(unittest.TestCase):
    def test_process_from_file(self):
        data_set_dict = {
            'yelp.p': ('test/data_for_tests/io/yelp_review_polarity', YelpPolarityPipe, (6, 6, 6), (1176, 2), False),
            'yelp.f': ('test/data_for_tests/io/yelp_review_full', YelpFullPipe, (6, 6, 6), (1023, 5), False),
            'sst-2': ('test/data_for_tests/io/SST-2', SST2Pipe, (5, 5, 5), (139, 2), True),
            'sst': ('test/data_for_tests/io/SST', SSTPipe, (6, 354, 6), (232, 5), False),
            'imdb': ('test/data_for_tests/io/imdb', IMDBPipe, (6, 6, 6), (1670, 2), False),
        }
        for k, v in data_set_dict.items():
            path, pipe, data_set, vocab, warns = v
            with self.subTest(pipe=pipe):
                if warns:
                    with self.assertWarns(Warning):
                        data_bundle = pipe(tokenizer='raw').process_from_file(path)
                else:
                    data_bundle = pipe(tokenizer='raw').process_from_file(path)

                self.assertTrue(isinstance(data_bundle, DataBundle))
                self.assertEqual(len(data_set), data_bundle.num_dataset)
                for x, y in zip(data_set, data_bundle.iter_datasets()):
                    name, dataset = y
                    self.assertEqual(x, len(dataset))

                self.assertEqual(len(vocab), data_bundle.num_vocab)
                for x, y in zip(vocab, data_bundle.iter_vocabs()):
                    name, vocabs = y
                    self.assertEqual(x, len(vocabs))


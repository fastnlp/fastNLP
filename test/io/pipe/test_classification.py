import unittest
import os

from fastNLP.io import DataBundle
from fastNLP.io.pipe.classification import SSTPipe, SST2Pipe, IMDBPipe, YelpFullPipe, YelpPolarityPipe, \
    AGsNewsPipe, DBPediaPipe
from fastNLP.io.pipe.classification import ChnSentiCorpPipe, THUCNewsPipe, WeiboSenti100kPipe


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
            'yelp.p': ('test/data_for_tests/io/yelp_review_polarity', YelpPolarityPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1176, 'target': 2},
                       False),
            'yelp.f': ('test/data_for_tests/io/yelp_review_full', YelpFullPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1166, 'target': 5},
                       False),
            'sst-2': ('test/data_for_tests/io/SST-2', SST2Pipe,
                      {'train': 5, 'dev': 5, 'test': 5}, {'words': 139, 'target': 2},
                      True),
            'sst': ('test/data_for_tests/io/SST', SSTPipe,
                    {'train': 354, 'dev': 6, 'test': 6}, {'words': 232, 'target': 5},
                    False),
            'imdb': ('test/data_for_tests/io/imdb', IMDBPipe,
                     {'train': 6, 'dev': 6, 'test': 6}, {'words': 1670, 'target': 2},
                     False),
            'ag': ('test/data_for_tests/io/ag', AGsNewsPipe,
                   {'train': 4, 'test': 5}, {'words': 257, 'target': 4},
                   False),
            'dbpedia': ('test/data_for_tests/io/dbpedia', DBPediaPipe,
                        {'train': 14, 'test': 5}, {'words': 496, 'target': 14},
                        False),
            'ChnSentiCorp': ('test/data_for_tests/io/ChnSentiCorp', ChnSentiCorpPipe,
                             {'train': 6, 'dev': 6, 'test': 6},
                             {'chars': 529, 'bigrams': 1296, 'trigrams': 1483, 'target': 2},
                             False),
            'Chn-THUCNews': ('test/data_for_tests/io/THUCNews', THUCNewsPipe,
                             {'train': 9, 'dev': 9, 'test': 9}, {'chars': 1864, 'target': 9},
                             False),
            'Chn-WeiboSenti100k': ('test/data_for_tests/io/WeiboSenti100k', WeiboSenti100kPipe,
                                   {'train': 6, 'dev': 6, 'test': 7}, {'chars': 452, 'target': 2},
                                   False),
        }
        for k, v in data_set_dict.items():
            path, pipe, data_set, vocab, warns = v
            with self.subTest(path=path):
                if 'Chn' not in k:
                    if warns:
                        with self.assertWarns(Warning):
                            data_bundle = pipe(tokenizer='raw').process_from_file(path)
                    else:
                        data_bundle = pipe(tokenizer='raw').process_from_file(path)
                else:
                    data_bundle = pipe(bigrams=True, trigrams=True).process_from_file(path)

                self.assertTrue(isinstance(data_bundle, DataBundle))
                self.assertEqual(len(data_set), data_bundle.num_dataset)
                for name, dataset in data_bundle.iter_datasets():
                    self.assertTrue(name in data_set.keys())
                    self.assertEqual(data_set[name], len(dataset))

                self.assertEqual(len(vocab), data_bundle.num_vocab)
                for name, vocabs in data_bundle.iter_vocabs():
                    self.assertTrue(name in vocab.keys())
                    self.assertEqual(vocab[name], len(vocabs))


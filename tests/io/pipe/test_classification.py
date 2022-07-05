import pytest
import os

from fastNLP.io import DataBundle
from fastNLP.io.pipe.classification import SSTPipe, SST2Pipe, IMDBPipe, YelpFullPipe, YelpPolarityPipe, \
    AGsNewsPipe, DBPediaPipe
from fastNLP.io.pipe.classification import ChnSentiCorpPipe, THUCNewsPipe, WeiboSenti100kPipe


@pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestClassificationPipe:
    def test_process_from_file(self):
        for pipe in [YelpPolarityPipe, SST2Pipe, IMDBPipe, YelpFullPipe, SSTPipe]:
            print(pipe)
            data_bundle = pipe(tokenizer='raw', num_proc=0).process_from_file()
            print(data_bundle)

    def test_process_from_file_proc(self, num_proc=2):
        for pipe in [YelpPolarityPipe, SST2Pipe, IMDBPipe, YelpFullPipe, SSTPipe]:
            print(pipe)
            data_bundle = pipe(tokenizer='raw', num_proc=num_proc).process_from_file()
            print(data_bundle)

class TestRunPipe:
    def test_load(self):
        for pipe in [IMDBPipe]:
            data_bundle = pipe(tokenizer='raw', num_proc=0).process_from_file('tests/data_for_tests/io/imdb')
            print(data_bundle)

    def test_load_proc(self):
        for pipe in [IMDBPipe]:
            data_bundle = pipe(tokenizer='raw', num_proc=2).process_from_file('tests/data_for_tests/io/imdb')
            print(data_bundle)


@pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestCNClassificationPipe:
    def test_process_from_file(self):
        for pipe in [ChnSentiCorpPipe]:
            data_bundle = pipe(bigrams=True, trigrams=True).process_from_file()
            print(data_bundle)


# @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestRunClassificationPipe:
    def test_process_from_file(self):
        data_set_dict = {
            'yelp.p': ('tests/data_for_tests/io/yelp_review_polarity', YelpPolarityPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1176, 'target': 2},
                       False),
            'yelp.f': ('tests/data_for_tests/io/yelp_review_full', YelpFullPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1166, 'target': 5},
                       False),
            'sst-2': ('tests/data_for_tests/io/SST-2', SST2Pipe,
                      {'train': 5, 'dev': 5, 'test': 5}, {'words': 139, 'target': 2},
                      True),
            'sst': ('tests/data_for_tests/io/SST', SSTPipe,
                    {'train': 354, 'dev': 6, 'test': 6}, {'words': 232, 'target': 5},
                    False),
            'imdb': ('tests/data_for_tests/io/imdb', IMDBPipe,
                     {'train': 6, 'dev': 6, 'test': 6}, {'words': 1670, 'target': 2},
                     False),
            'ag': ('tests/data_for_tests/io/ag', AGsNewsPipe,
                   {'train': 4, 'test': 5}, {'words': 257, 'target': 4},
                   False),
            'dbpedia': ('tests/data_for_tests/io/dbpedia', DBPediaPipe,
                        {'train': 14, 'test': 5}, {'words': 496, 'target': 14},
                        False),
            'ChnSentiCorp': ('tests/data_for_tests/io/ChnSentiCorp', ChnSentiCorpPipe,
                             {'train': 6, 'dev': 6, 'test': 6},
                             {'chars': 529, 'bigrams': 1296, 'trigrams': 1483, 'target': 2},
                             False),
            'Chn-THUCNews': ('tests/data_for_tests/io/THUCNews', THUCNewsPipe,
                             {'train': 9, 'dev': 9, 'test': 9}, {'chars': 1864, 'target': 9},
                             False),
            'Chn-WeiboSenti100k': ('tests/data_for_tests/io/WeiboSenti100k', WeiboSenti100kPipe,
                                   {'train': 6, 'dev': 6, 'test': 7}, {'chars': 452, 'target': 2},
                                   False),
        }
        for k, v in data_set_dict.items():
            path, pipe, data_set, vocab, warns = v
            if 'Chn' not in k:
                if warns:
                    data_bundle = pipe(tokenizer='raw', num_proc=0).process_from_file(path)
                else:
                    data_bundle = pipe(tokenizer='raw', num_proc=0).process_from_file(path)
            else:
                data_bundle = pipe(bigrams=True, trigrams=True).process_from_file(path)

            assert(isinstance(data_bundle, DataBundle))
            assert(len(data_set) == data_bundle.num_dataset)
            for name, dataset in data_bundle.iter_datasets():
                assert(name in data_set.keys())
                assert(data_set[name] == len(dataset))

            assert(len(vocab) == data_bundle.num_vocab)
            for name, vocabs in data_bundle.iter_vocabs():
                assert(name in vocab.keys())
                assert(vocab[name] == len(vocabs))

    def test_process_from_file_proc(self):
        data_set_dict = {
            'yelp.p': ('tests/data_for_tests/io/yelp_review_polarity', YelpPolarityPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1176, 'target': 2},
                       False),
            'yelp.f': ('tests/data_for_tests/io/yelp_review_full', YelpFullPipe,
                       {'train': 6, 'dev': 6, 'test': 6}, {'words': 1166, 'target': 5},
                       False),
            'sst-2': ('tests/data_for_tests/io/SST-2', SST2Pipe,
                      {'train': 5, 'dev': 5, 'test': 5}, {'words': 139, 'target': 2},
                      True),
            'sst': ('tests/data_for_tests/io/SST', SSTPipe,
                    {'train': 354, 'dev': 6, 'test': 6}, {'words': 232, 'target': 5},
                    False),
            'imdb': ('tests/data_for_tests/io/imdb', IMDBPipe,
                     {'train': 6, 'dev': 6, 'test': 6}, {'words': 1670, 'target': 2},
                     False),
            'ag': ('tests/data_for_tests/io/ag', AGsNewsPipe,
                   {'train': 4, 'test': 5}, {'words': 257, 'target': 4},
                   False),
            'dbpedia': ('tests/data_for_tests/io/dbpedia', DBPediaPipe,
                        {'train': 14, 'test': 5}, {'words': 496, 'target': 14},
                        False),
            'ChnSentiCorp': ('tests/data_for_tests/io/ChnSentiCorp', ChnSentiCorpPipe,
                             {'train': 6, 'dev': 6, 'test': 6},
                             {'chars': 529, 'bigrams': 1296, 'trigrams': 1483, 'target': 2},
                             False),
            'Chn-THUCNews': ('tests/data_for_tests/io/THUCNews', THUCNewsPipe,
                             {'train': 9, 'dev': 9, 'test': 9}, {'chars': 1864, 'target': 9},
                             False),
            'Chn-WeiboSenti100k': ('tests/data_for_tests/io/WeiboSenti100k', WeiboSenti100kPipe,
                                   {'train': 6, 'dev': 6, 'test': 7}, {'chars': 452, 'target': 2},
                                   False),
        }
        for k, v in data_set_dict.items():
            path, pipe, data_set, vocab, warns = v
            if 'Chn' not in k:
                if warns:
                    data_bundle = pipe(tokenizer='raw', num_proc=2).process_from_file(path)
                else:
                    data_bundle = pipe(tokenizer='raw', num_proc=2).process_from_file(path)
            else:
                # if k == 'ChnSentiCorp':
                #     data_bundle = pipe(bigrams=True, trigrams=True).process_from_file(path)
                # else:
                data_bundle = pipe(bigrams=True, trigrams=True, num_proc=2).process_from_file(path)

            assert(isinstance(data_bundle, DataBundle))
            assert(len(data_set) == data_bundle.num_dataset)
            for name, dataset in data_bundle.iter_datasets():
                assert(name in data_set.keys())
                assert(data_set[name] == len(dataset))

            assert(len(vocab) == data_bundle.num_vocab)
            for name, vocabs in data_bundle.iter_vocabs():
                assert(name in vocab.keys())
                assert(vocab[name] == len(vocabs))
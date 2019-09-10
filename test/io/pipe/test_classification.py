import unittest
import os

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
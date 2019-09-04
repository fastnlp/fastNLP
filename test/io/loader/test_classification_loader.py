
import unittest
from fastNLP.io.loader.classification import YelpFullLoader
from fastNLP.io.loader.classification import YelpPolarityLoader
from fastNLP.io.loader.classification import IMDBLoader
from fastNLP.io.loader.classification import SST2Loader
from fastNLP.io.loader.classification import SSTLoader
from fastNLP.io.loader.classification import ChnSentiCorpLoader
import os

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestDownload(unittest.TestCase):
    def test_download(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            loader().download()

    def test_load(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            data_bundle = loader().load()
            print(data_bundle)


class TestLoad(unittest.TestCase):
    def test_load(self):
        for loader in [IMDBLoader]:
            data_bundle = loader().load('test/data_for_tests/io/imdb')
            print(data_bundle)

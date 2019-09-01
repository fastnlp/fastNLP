
import unittest
from fastNLP.io.loader.matching import RTELoader
from fastNLP.io.loader.matching import QNLILoader
from fastNLP.io.loader.matching import SNLILoader
from fastNLP.io.loader.matching import QuoraLoader
from fastNLP.io.loader.matching import MNLILoader
import os

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestDownload(unittest.TestCase):
    def test_download(self):
        for loader in [RTELoader, QNLILoader, SNLILoader, MNLILoader]:
            loader().download()
        with self.assertRaises(Exception):
            QuoraLoader().load()

    def test_load(self):
        for loader in [RTELoader, QNLILoader, SNLILoader, MNLILoader]:
            data_bundle = loader().load()
            print(data_bundle)


class TestLoad(unittest.TestCase):

    def test_load(self):
        for loader in [RTELoader]:
            data_bundle = loader().load('test/data_for_tests/io/rte')
            print(data_bundle)


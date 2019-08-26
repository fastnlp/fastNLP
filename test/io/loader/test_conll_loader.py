
import unittest
import os
from fastNLP.io.loader.conll import MsraNERLoader, PeopleDailyNERLoader, WeiboNERLoader

class MSRANERTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        MsraNERLoader().download(re_download=False)
        data_bundle = MsraNERLoader().load()
        print(data_bundle)

class PeopleDailyTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        PeopleDailyNERLoader().download()

class WeiboNERTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        WeiboNERLoader().download()
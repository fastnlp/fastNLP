
import unittest
import os
from fastNLP.io.loader.conll import MsraNERLoader, PeopleDailyNERLoader, WeiboNERLoader, \
    Conll2003Loader


class TestMSRANER(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        MsraNERLoader().download(re_download=False)
        data_bundle = MsraNERLoader().load()
        print(data_bundle)


class TestPeopleDaily(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        PeopleDailyNERLoader().download()


class TestWeiboNER(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        WeiboNERLoader().download()


class TestConll2003Loader(unittest.TestCase):
    def test__load(self):
        Conll2003Loader()._load('test/data_for_tests/conll_2003_example.txt')


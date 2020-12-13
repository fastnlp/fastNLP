
import unittest
import os
from fastNLP.io.loader.conll import MsraNERLoader, PeopleDailyNERLoader, WeiboNERLoader, \
    Conll2003Loader, ConllLoader


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
    def test_load(self):
        Conll2003Loader()._load('tests/data_for_tests/conll_2003_example.txt')


class TestConllLoader(unittest.TestCase):
    def test_conll(self):
        db = Conll2003Loader().load('tests/data_for_tests/io/conll2003')
        print(db)

class TestConllLoader(unittest.TestCase):
    def test_sep(self):
        headers = [
            'raw_words',  'ner',
        ]
        db = ConllLoader(headers = headers,sep="\n").load('tests/data_for_tests/io/MSRA_NER')
        print(db)

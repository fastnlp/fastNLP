
import pytest
import os
from fastNLP.io.loader.conll import MsraNERLoader, PeopleDailyNERLoader, WeiboNERLoader, \
    Conll2003Loader, ConllLoader


class TestMSRANER:
    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_download(self):
        MsraNERLoader().download(re_download=False)
        data_bundle = MsraNERLoader().load()
        print(data_bundle)


class TestPeopleDaily:
    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_download(self):
        PeopleDailyNERLoader().download()


class TestWeiboNER:
    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_download(self):
        WeiboNERLoader().download()


class TestConll2003Loader:
    def test_load(self):
        Conll2003Loader()._load('data_for_tests/conll_2003_example.txt')


class TestConllLoader:
    def test_conll(self):
        db = Conll2003Loader().load('data_for_tests/io/conll2003')
        print(db)

    def test_sep(self):
        headers = [
            'raw_words',  'ner',
        ]
        db = ConllLoader(headers = headers, sep="\n").load('data_for_tests/io/MSRA_NER')
        print(db)

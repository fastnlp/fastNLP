import unittest
import os
from fastNLP.io import MsraNERPipe, PeopleDailyPipe, WeiboNERPipe, Conll2003Pipe, Conll2003NERPipe, \
    OntoNotesNERPipe


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestConllPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [MsraNERPipe, PeopleDailyPipe, WeiboNERPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(bigrams=True, trigrams=True).process_from_file()
                print(data_bundle)
                data_bundle = pipe(encoding_type='bioes').process_from_file()
                print(data_bundle)


class TestRunPipe(unittest.TestCase):
    def test_conll2003(self):
        for pipe in [Conll2003Pipe, Conll2003NERPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe().process_from_file('tests/data_for_tests/conll_2003_example.txt')
                print(data_bundle)


class TestNERPipe(unittest.TestCase):
    def test_process_from_file(self):
        data_dict = {
            'weibo_NER': WeiboNERPipe,
            'peopledaily': PeopleDailyPipe,
            'MSRA_NER': MsraNERPipe,
        }
        for k, v in data_dict.items():
            pipe = v
            with self.subTest(pipe=pipe):
                data_bundle = pipe(bigrams=True, trigrams=True).process_from_file(f'tests/data_for_tests/io/{k}')
                print(data_bundle)
                data_bundle = pipe(encoding_type='bioes').process_from_file(f'tests/data_for_tests/io/{k}')
                print(data_bundle)


class TestConll2003Pipe(unittest.TestCase):
    def test_conll(self):
        with self.assertWarns(Warning):
            data_bundle = Conll2003Pipe().process_from_file('tests/data_for_tests/io/conll2003')
        print(data_bundle)

    def test_OntoNotes(self):
        data_bundle = OntoNotesNERPipe().process_from_file('tests/data_for_tests/io/OntoNotes')
        print(data_bundle)

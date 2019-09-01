import unittest
import os
from fastNLP.io import MsraNERPipe, PeopleDailyPipe, WeiboNERPipe, Conll2003Pipe, Conll2003NERPipe


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [MsraNERPipe, PeopleDailyPipe, WeiboNERPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe().process_from_file()
                print(data_bundle)


class TestRunPipe(unittest.TestCase):
    def test_conll2003(self):
        for pipe in [Conll2003Pipe, Conll2003NERPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe().process_from_file('test/data_for_tests/conll_2003_example.txt')
                print(data_bundle)

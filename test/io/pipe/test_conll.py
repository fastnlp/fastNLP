import unittest
import os
from fastNLP.io import MsraNERPipe, PeopleDailyPipe, WeiboNERPipe

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [MsraNERPipe, PeopleDailyPipe, WeiboNERPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe().process_from_file()
                print(data_bundle)
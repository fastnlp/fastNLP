
import unittest
import os

from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, QNLIPipe, MNLIPipe
from fastNLP.io.pipe.matching import SNLIBertPipe, RTEBertPipe, QNLIBertPipe, MNLIBertPipe


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [SNLIPipe, RTEPipe, QNLIPipe, MNLIPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestBertPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [SNLIBertPipe, RTEBertPipe, QNLIBertPipe, MNLIBertPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)


class TestRunPipe(unittest.TestCase):

    def test_load(self):
        for pipe in [RTEPipe, RTEBertPipe]:
            data_bundle = pipe(tokenizer='raw').process_from_file('test/data_for_tests/io/rte')
            print(data_bundle)

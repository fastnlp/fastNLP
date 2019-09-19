
import unittest
import os
from fastNLP.io.pipe.cws import CWSPipe


class TestCWSPipe(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_process_from_file(self):
        dataset_names = ['pku', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSPipe(dataset_name=dataset_name).process_from_file()
                print(data_bundle)


class TestRunCWSPipe(unittest.TestCase):
    def test_process_from_file(self):
        dataset_names = ['msra', 'cityu', 'as', 'pku']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSPipe(bigrams=True, trigrams=True).\
                    process_from_file(f'test/data_for_tests/io/cws_{dataset_name}')
                print(data_bundle)

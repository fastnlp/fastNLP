import unittest
import os
from fastNLP.io.loader import CWSLoader


class CWSLoaderTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_download(self):
        dataset_names = ['pku', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSLoader(dataset_name=dataset_name).load()
                print(data_bundle)
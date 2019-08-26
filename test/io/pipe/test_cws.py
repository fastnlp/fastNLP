
import unittest
import os
from fastNLP.io.pipe.cws import CWSPipe

class CWSPipeTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_process_from_file(self):
        dataset_names = ['pku', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSPipe(dataset_name=dataset_name).process_from_file()
                print(data_bundle)
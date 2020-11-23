import unittest

from fastNLP.io.loader.qa import CMRC2018Loader

class TestCMRC2018Loader(unittest.TestCase):
    def test__load(self):
        loader = CMRC2018Loader()
        dataset = loader._load('tests/data_for_tests/io/cmrc/train.json')
        print(dataset)

    def test_load(self):
        loader = CMRC2018Loader()
        data_bundle = loader.load('tests/data_for_tests/io/cmrc/')
        print(data_bundle)

import unittest
from reproduction.text_classification.data.MTL16Loader import MTL16Loader


class TestDataLoader(unittest.TestCase):
    def test_MTL16Loader(self):
        loader = MTL16Loader()
        data = loader.process('sample_MTL16.txt')
        print(data.datasets)


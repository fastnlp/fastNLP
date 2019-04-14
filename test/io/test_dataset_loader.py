import unittest

from fastNLP.io.dataset_loader import Conll2003Loader, PeopleDailyCorpusLoader, ConllCWSReader, \
    ZhConllPOSReader, ConllxDataLoader


class TestDatasetLoader(unittest.TestCase):

    def test_Conll2003Loader(self):
        """
            Test the the loader of Conll2003 dataset
        """
        dataset_path = "test/data_for_tests/conll_2003_example.txt"
        loader = Conll2003Loader()
        dataset_2003 = loader.load(dataset_path)

    def test_PeopleDailyCorpusLoader(self):
        data_set = PeopleDailyCorpusLoader().load("test/data_for_tests/people_daily_raw.txt")


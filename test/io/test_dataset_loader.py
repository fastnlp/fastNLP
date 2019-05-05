import unittest

from fastNLP.io.dataset_loader import Conll2003Loader, PeopleDailyCorpusLoader, \
    CSVLoader, SNLILoader, JsonLoader

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

    def test_CSVLoader(self):
        ds = CSVLoader(sep='\t', headers=['words', 'label'])\
            .load('test/data_for_tests/tutorial_sample_dataset.csv')
        assert len(ds) > 0

    def test_SNLILoader(self):
        ds = SNLILoader().load('test/data_for_tests/sample_snli.jsonl')
        assert len(ds) == 3

    def test_JsonLoader(self):
        ds = JsonLoader().load('test/data_for_tests/sample_snli.jsonl')
        assert len(ds) == 3


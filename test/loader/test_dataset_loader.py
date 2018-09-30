import os
import unittest

from fastNLP.loader.dataset_loader import POSDataSetLoader, LMDataSetLoader, TokenizeDataSetLoader, \
    PeopleDailyCorpusLoader, ConllLoader


class TestDatasetLoader(unittest.TestCase):
    def test_case_1(self):
        data = """Tom\tT\nand\tF\nJerry\tT\n.\tF\n\nHello\tT\nworld\tF\n!\tF"""
        lines = data.split("\n")
        answer = POSDataSetLoader.parse(lines)
        truth = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]], [["Hello", "world", "!"], ["T", "F", "F"]]]
        self.assertListEqual(answer, truth, "POS Dataset Loader")

    def test_case_TokenizeDatasetLoader(self):
        loader = TokenizeDataSetLoader()
        data = loader.load("./test/data_for_tests/cws_pku_utf_8", max_seq_len=32)
        print("pass TokenizeDataSetLoader test!")

    def test_case_POSDatasetLoader(self):
        loader = POSDataSetLoader()
        data = loader.load("./test/data_for_tests/people.txt")
        datas = loader.load_lines("./test/data_for_tests/people.txt")
        print("pass POSDataSetLoader test!")

    def test_case_LMDatasetLoader(self):
        loader = LMDataSetLoader()
        data = loader.load("./test/data_for_tests/charlm.txt")
        datas = loader.load_lines("./test/data_for_tests/charlm.txt")
        print("pass TokenizeDataSetLoader test!")

    def test_PeopleDailyCorpusLoader(self):
        loader = PeopleDailyCorpusLoader()
        _, _ = loader.load("./test/data_for_tests/people_daily_raw.txt")

    def test_ConllLoader(self):
        loader = ConllLoader()
        _ = loader.load("./test/data_for_tests/conll_example.txt")


if __name__ == '__main__':
    unittest.main()

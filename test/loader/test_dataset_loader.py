import unittest

from fastNLP.loader.dataset_loader import POSDatasetLoader, LMDatasetLoader, TokenizeDatasetLoader, \
    PeopleDailyCorpusLoader, ConllLoader


class TestDatasetLoader(unittest.TestCase):
    def test_case_1(self):
        data = """Tom\tT\nand\tF\nJerry\tT\n.\tF\n\nHello\tT\nworld\tF\n!\tF"""
        lines = data.split("\n")
        answer = POSDatasetLoader.parse(lines)
        truth = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]], [["Hello", "world", "!"], ["T", "F", "F"]]]
        self.assertListEqual(answer, truth, "POS Dataset Loader")

    def test_case_TokenizeDatasetLoader(self):
        loader = TokenizeDatasetLoader("./test/data_for_tests/cws_pku_utf_8")
        data = loader.load_pku(max_seq_len=32)
        print("pass TokenizeDatasetLoader test!")

    def test_case_POSDatasetLoader(self):
        loader = POSDatasetLoader("./test/data_for_tests/people.txt")
        data = loader.load()
        datas = loader.load_lines()
        print("pass POSDatasetLoader test!")

    def test_case_LMDatasetLoader(self):
        loader = LMDatasetLoader("./test/data_for_tests/cws_pku_utf_8")
        data = loader.load()
        datas = loader.load_lines()
        print("pass TokenizeDatasetLoader test!")

    def test_PeopleDailyCorpusLoader(self):
        loader = PeopleDailyCorpusLoader("./test/data_for_tests/people_daily_raw.txt")
        _, _ = loader.load()

    def test_ConllLoader(self):
        loader = ConllLoader("./test/data_for_tests/conll_example.txt")
        _ = loader.load()


if __name__ == '__main__':
    unittest.main()

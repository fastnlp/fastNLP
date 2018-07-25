import unittest

from fastNLP.loader.dataset_loader import POSDatasetLoader


class TestPreprocess(unittest.TestCase):
    def test_case_1(self):
        data = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]],
                ["Hello", "world", "!"], ["T", "F", "F"]]
        pickle_path = "./data_for_tests/"
        # POSPreprocess(data, pickle_path)


class TestDatasetLoader(unittest.TestCase):
    def test_case_1(self):
        data = """Tom\tT\nand\tF\nJerry\tT\n.\tF\n\nHello\tT\nworld\tF\n!\tF"""
        lines = data.split("\n")
        answer = POSDatasetLoader.parse(lines)
        truth = [[["Tom", "and", "Jerry", "."], ["T", "F", "T", "F"]], [["Hello", "world", "!"], ["T", "F", "F"]]]
        self.assertListEqual(answer, truth, "POS Dataset Loader")


if __name__ == '__main__':
    unittest.main()

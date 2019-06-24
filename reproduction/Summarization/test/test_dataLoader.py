

import unittest
from ..data.dataloader import SummarizationLoader


class TestSummarizationLoader(unittest.TestCase):
    def test_case1(self):
        sum_loader = SummarizationLoader()
        paths = {"train":"testdata/train.jsonl", "valid":"testdata/val.jsonl", "test":"testdata/test.jsonl"}
        data = sum_loader.process(paths=paths)
        print(data.datasets)

    def test_case2(self):
        sum_loader = SummarizationLoader()
        paths = {"train": "testdata/train.jsonl", "valid": "testdata/val.jsonl", "test": "testdata/test.jsonl"}
        data = sum_loader.process(paths=paths, domain=True)
        print(data.datasets, data.vocabs)

    def test_case3(self):
        sum_loader = SummarizationLoader()
        paths = {"train": "testdata/train.jsonl", "valid": "testdata/val.jsonl", "test": "testdata/test.jsonl"}
        data = sum_loader.process(paths=paths, tag=True)
        print(data.datasets, data.vocabs)
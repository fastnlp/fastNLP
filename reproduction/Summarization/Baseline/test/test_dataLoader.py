
import unittest

import sys
sys.path.append('..')

from data.dataloader import SummarizationLoader

vocab_size = 100000
vocab_path = "testdata/vocab"
sent_max_len = 100
doc_max_timesteps = 50

class TestSummarizationLoader(unittest.TestCase):

    def test_case1(self):
        sum_loader = SummarizationLoader()
        paths = {"train":"testdata/train.jsonl", "valid":"testdata/val.jsonl", "test":"testdata/test.jsonl"}
        data = sum_loader.process(paths=paths, vocab_size=vocab_size, vocab_path=vocab_path, sent_max_len=sent_max_len, doc_max_timesteps=doc_max_timesteps)
        print(data.datasets)

    def test_case2(self):
        sum_loader = SummarizationLoader()
        paths = {"train": "testdata/train.jsonl", "valid": "testdata/val.jsonl", "test": "testdata/test.jsonl"}
        data = sum_loader.process(paths=paths, vocab_size=vocab_size, vocab_path=vocab_path, sent_max_len=sent_max_len, doc_max_timesteps=doc_max_timesteps, domain=True)
        print(data.datasets, data.vocabs)

    def test_case3(self):
        sum_loader = SummarizationLoader()
        paths = {"train": "testdata/train.jsonl", "valid": "testdata/val.jsonl", "test": "testdata/test.jsonl"}
        data = sum_loader.process(paths=paths, vocab_size=vocab_size, vocab_path=vocab_path, sent_max_len=sent_max_len, doc_max_timesteps=doc_max_timesteps, tag=True)
        print(data.datasets, data.vocabs)





import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import unittest
import torch
from fastNLP.data.field import TextField, LabelField
from fastNLP.data.instance import Instance
from fastNLP.data.dataset import DataSet
from fastNLP.data.batch import Batch



class TestField(unittest.TestCase):
    def check_batched_data_equal(self, data1, data2):
        self.assertEqual(len(data1), len(data2))
        for i in range(len(data1)):
            self.assertTrue(data1[i].keys(), data2[i].keys())
        for i in range(len(data1)):
            for t1, t2 in zip(data1[i].values(), data2[i].values()):
                self.assertTrue(torch.equal(t1, t2))
    
    def test_batchiter(self):
        texts = [
            "i am a cat",
            "this is a test of new batch",
            "haha"
        ]
        labels = [0, 1, 0]
        
        # prepare vocabulary
        vocab = {}
        for text in texts:
            for tokens in text.split():
                if tokens not in vocab:
                    vocab[tokens] = len(vocab)

        # prepare input dataset    
        data = DataSet()
        for text, label in zip(texts, labels):
            x = TextField(text.split(), False)
            y = LabelField(label, is_target=True)
            ins = Instance(text=x, label=y)
            data.append(ins)
        
        # use vocabulary to index data
        data.index_field("text", vocab)

        # define naive sampler for batch class
        class SeqSampler:
            def __call__(self, dataset):
                return list(range(len(dataset)))
                
        # use bacth to iterate dataset
        batcher = Batch(data, SeqSampler(), 2)
        TRUE_X = [{'text': torch.tensor([[0, 1, 2, 3, 0, 0, 0], [4, 5, 2, 6, 7, 8, 9]])}, {'text': torch.tensor([[10]])}]
        TRUE_Y = [{'label': torch.tensor([[0], [1]])}, {'label': torch.tensor([[0]])}]
        for epoch in range(3):
            test_x, test_y = [], []
            for batch_x, batch_y in batcher:
                test_x.append(batch_x)
                test_y.append(batch_y)
            self.check_batched_data_equal(TRUE_X, test_x)
            self.check_batched_data_equal(TRUE_Y, test_y)


if __name__ == "__main__":
    unittest.main()
    
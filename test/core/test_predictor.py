import os
import unittest

from fastNLP.core.predictor import Predictor
from fastNLP.core.preprocess import save_pickle
from fastNLP.models.sequence_modeling import SeqLabeling


class TestPredictor(unittest.TestCase):
    def test_seq_label(self):
        model_args = {
            "vocab_size": 10,
            "word_emb_dim": 100,
            "rnn_hidden_units": 100,
            "num_classes": 5
        }

        infer_data = [
            ['a', 'b', 'c', 'd', 'e'],
            ['a', '@', 'c', 'd', 'e'],
            ['a', 'b', '#', 'd', 'e'],
            ['a', 'b', 'c', '?', 'e'],
            ['a', 'b', 'c', 'd', '$'],
            ['!', 'b', 'c', 'd', 'e']
        ]
        vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, '!': 5, '@': 6, '#': 7, '$': 8, '?': 9}

        os.system("mkdir save")
        save_pickle({0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}, "./save/", "id2class.pkl")
        save_pickle(vocab, "./save/", "word2id.pkl")

        model = SeqLabeling(model_args)
        predictor = Predictor("./save/", task="seq_label")

        results = predictor.predict(network=model, data=infer_data)

        self.assertTrue(isinstance(results, list))
        self.assertGreater(len(results), 0)
        for res in results:
            self.assertTrue(isinstance(res, list))
            self.assertEqual(len(res), 5)
            self.assertTrue(isinstance(res[0], str))

        os.system("rm -rf save")
        print("pickle path deleted")


class TestPredictor2(unittest.TestCase):
    def test_text_classify(self):
        # TODO
        pass

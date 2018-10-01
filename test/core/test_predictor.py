import os
import unittest

from fastNLP.core.dataset import TextClassifyDataSet, SeqLabelDataSet
from fastNLP.core.predictor import Predictor
from fastNLP.core.preprocess import save_pickle
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.loader.base_loader import BaseLoader
from fastNLP.models.cnn_text_classification import CNNText
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

        vocab = Vocabulary()
        vocab.word2idx = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, '!': 5, '@': 6, '#': 7, '$': 8, '?': 9}
        class_vocab = Vocabulary()
        class_vocab.word2idx = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

        os.system("mkdir save")
        save_pickle(class_vocab, "./save/", "label2id.pkl")
        save_pickle(vocab, "./save/", "word2id.pkl")

        model = CNNText(model_args)
        import fastNLP.core.predictor as pre
        predictor = Predictor("./save/", pre.text_classify_post_processor)

        # Load infer data
        infer_data_set = TextClassifyDataSet(load_func=BaseLoader.load)
        infer_data_set.convert_for_infer(infer_data, vocabs={"word_vocab": vocab.word2idx})

        results = predictor.predict(network=model, data=infer_data_set)

        self.assertTrue(isinstance(results, list))
        self.assertGreater(len(results), 0)
        self.assertEqual(len(results), len(infer_data))
        for res in results:
            self.assertTrue(isinstance(res, str))
            self.assertTrue(res in class_vocab.word2idx)

        del model, predictor, infer_data_set

        model = SeqLabeling(model_args)
        predictor = Predictor("./save/", pre.seq_label_post_processor)

        infer_data_set = SeqLabelDataSet(load_func=BaseLoader.load)
        infer_data_set.convert_for_infer(infer_data, vocabs={"word_vocab": vocab.word2idx})

        results = predictor.predict(network=model, data=infer_data_set)
        self.assertTrue(isinstance(results, list))
        self.assertEqual(len(results), len(infer_data))
        for i in range(len(infer_data)):
            res = results[i]
            self.assertTrue(isinstance(res, list))
            self.assertEqual(len(res), len(infer_data[i]))

        os.system("rm -rf save")
        print("pickle path deleted")


class TestPredictor2(unittest.TestCase):
    def test_text_classify(self):
        # TODO
        pass

import unittest

from fastNLP.embeddings import StaticEmbedding
from fastNLP import Vocabulary
import torch

class TestRandomSameEntry(unittest.TestCase):
    def test_same_vector(self):
        vocab = Vocabulary().add_word_lst(["The", "the", "THE"])
        embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE"]]])
        words = embed(words)
        embed_0 = words[0, 0]
        for i in range(1, words.size(1)):
            assert torch.sum(embed_0==words[0, i]).eq(len(embed_0))

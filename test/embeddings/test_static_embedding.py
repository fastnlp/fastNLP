import unittest

from fastNLP.embeddings import StaticEmbedding
from fastNLP import Vocabulary
import torch
import os

class TestRandomSameEntry(unittest.TestCase):
    def test_same_vector(self):
        vocab = Vocabulary().add_word_lst(["The", "the", "THE", 'a', "A"])
        embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE", 'a', 'A']]])
        words = embed(words)
        embed_0 = words[0, 0]
        for i in range(1, 3):
            assert torch.sum(embed_0==words[0, i]).eq(len(embed_0))
        embed_0 = words[0, 3]
        for i in range(3, 5):
            assert torch.sum(embed_0 == words[0, i]).eq(len(embed_0))

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_same_vector2(self):
        vocab = Vocabulary().add_word_lst(["The", 'a', 'b', "the", "THE", "B", 'a', "A"])
        embed = StaticEmbedding(vocab, model_dir_or_name='/remote-home/source/fastnlp_caches/glove.6B.100d/glove.6B.100d.txt',
                                lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE", 'b', "B", 'a', 'A']]])
        words = embed(words)
        embed_0 = words[0, 0]
        for i in range(1, 3):
            assert torch.sum(embed_0==words[0, i]).eq(len(embed_0))
        embed_0 = words[0, 3]
        for i in range(3, 5):
            assert torch.sum(embed_0 == words[0, i]).eq(len(embed_0))

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_same_vector3(self):
        word_lst = ["The", "the"]
        no_create_word_lst = ['of', 'Of', 'With', 'with']
        vocab = Vocabulary().add_word_lst(word_lst)
        vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='/remote-home/source/fastnlp_caches/glove.6B.100d/glove.demo.txt',
                                lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in word_lst+no_create_word_lst]])
        words = embed(words)

        lowered_word_lst = [word.lower() for word in word_lst]
        lowered_no_create_word_lst = [word.lower() for word in no_create_word_lst]
        lowered_vocab = Vocabulary().add_word_lst(lowered_word_lst)
        lowered_vocab.add_word_lst(lowered_no_create_word_lst, no_create_entry=True)
        lowered_embed = StaticEmbedding(lowered_vocab, model_dir_or_name='/remote-home/source/fastnlp_caches/glove.6B.100d/glove.demo.txt',
                                lower=False)
        lowered_words = torch.LongTensor([[lowered_vocab.to_index(word) for word in lowered_word_lst+lowered_no_create_word_lst]])
        lowered_words = lowered_embed(lowered_words)

        all_words = word_lst + no_create_word_lst

        for idx, (word_i, word_j) in enumerate(zip(words[0], lowered_words[0])):
            with self.subTest(idx=idx, word=all_words[idx]):
                assert torch.sum(word_i == word_j).eq(lowered_embed.embed_size)

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_same_vector4(self):
        # words = []
        # create_word_lst = []  # 需要创建
        # no_create_word_lst = []
        # ignore_word_lst = []
        # with open('/remote-home/source/fastnlp_caches/glove.6B.100d/glove.demo.txt', 'r', encoding='utf-8') as f:
        #     for line in f:
        #         words
        word_lst = ["The", "the", "the", "The", "a", "A"]
        no_create_word_lst = ['of', 'Of', "Of", "of", 'With', 'with']
        all_words = word_lst[:-2] + no_create_word_lst[:-2]
        vocab = Vocabulary(min_freq=2).add_word_lst(word_lst)
        vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='/remote-home/source/fastnlp_caches/glove.6B.100d/glove.demo.txt',
                                lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in all_words]])
        words = embed(words)

        lowered_word_lst = [word.lower() for word in word_lst]
        lowered_no_create_word_lst = [word.lower() for word in no_create_word_lst]
        lowered_vocab = Vocabulary().add_word_lst(lowered_word_lst)
        lowered_vocab.add_word_lst(lowered_no_create_word_lst, no_create_entry=True)
        lowered_embed = StaticEmbedding(lowered_vocab, model_dir_or_name='/remote-home/source/fastnlp_caches/glove.6B.100d/glove.demo.txt',
                                lower=False)
        lowered_words = torch.LongTensor([[lowered_vocab.to_index(word.lower()) for word in all_words]])
        lowered_words = lowered_embed(lowered_words)

        for idx in range(len(all_words)):
            word_i, word_j = words[0, idx], lowered_words[0, idx]
            with self.subTest(idx=idx, word=all_words[idx]):
                assert torch.sum(word_i == word_j).eq(lowered_embed.embed_size)
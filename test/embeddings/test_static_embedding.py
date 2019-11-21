import unittest

from fastNLP.embeddings import StaticEmbedding
from fastNLP import Vocabulary
import torch
import os


class TestLoad(unittest.TestCase):
    def test_norm1(self):
        # 测试只对可以找到的norm
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_norm_found_vector=True)
        self.assertEqual(round(torch.norm(embed(torch.LongTensor([[2]]))).item(), 4), 1)
        self.assertNotEqual(torch.norm(embed(torch.LongTensor([[4]]))).item(), 1)

    def test_norm2(self):
        # 测试对所有都norm
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                normalize=True)
        self.assertEqual(round(torch.norm(embed(torch.LongTensor([[2]]))).item(), 4), 1)
        self.assertEqual(round(torch.norm(embed(torch.LongTensor([[4]]))).item(), 4), 1)

    def test_dropword(self):
        # 测试是否可以通过drop word
        vocab = Vocabulary().add_word_lst([chr(i) for i in range(1, 200)])
        embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=10, dropout=0.1, word_dropout=0.4)
        for i in range(10):
            length = torch.randint(1, 50, (1,)).item()
            batch = torch.randint(1, 4, (1,)).item()
            words = torch.randint(1, 200, (batch, length)).long()
            embed(words)

    def test_only_use_pretrain_word(self):
        def check_word_unk(words, vocab, embed):
            for word in words:
                self.assertListEqual(embed(torch.LongTensor([vocab.to_index(word)])).tolist()[0],
                                         embed(torch.LongTensor([1])).tolist()[0])

        def check_vector_equal(words, vocab, embed, embed_dict, lower=False):
            for word in words:
                index = vocab.to_index(word)
                v1 = embed(torch.LongTensor([index])).tolist()[0]
                if lower:
                    word = word.lower()
                v2 = embed_dict[word]
                for v1i, v2i in zip(v1, v2):
                    self.assertAlmostEqual(v1i, v2i, places=4)
        embed_dict = read_static_embed('test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt')

        # 测试是否只使用pretrain的word
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        vocab.add_word('of', no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True)
        #   notinfile应该被置为unk
        check_vector_equal(['the', 'a', 'of'], vocab, embed, embed_dict)
        check_word_unk(['notinfile'], vocab, embed)

        # 测试在大小写情况下的使用
        vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile'])
        vocab.add_word('Of', no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True)
        check_word_unk(['The', 'Of', 'notinfile'], vocab, embed)  # 这些词应该找不到
        check_vector_equal(['a'], vocab, embed, embed_dict)

        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True, lower=True)
        check_vector_equal(['The', 'Of', 'a'], vocab, embed, embed_dict, lower=True)
        check_word_unk(['notinfile'], vocab, embed)

        # 测试min_freq
        vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile1', 'A', 'notinfile2', 'notinfile2'])
        vocab.add_word('Of', no_create_entry=True)

        embed = StaticEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True, lower=True, min_freq=2, only_train_min_freq=True)

        check_vector_equal(['Of', 'a'], vocab, embed, embed_dict, lower=True)
        check_word_unk(['notinfile1', 'The', 'notinfile2'], vocab, embed)


def read_static_embed(fp):
    """

    :param str fp: embedding的路径
    :return: {}, key是word, value是vector
    """
    embed = {}
    with open(fp, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                vector = list(map(float, parts[1:]))
                word = parts[0]
                embed[word] = vector
    return embed


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
        embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6B-100d',
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
        # 验证lower
        word_lst = ["The", "the"]
        no_create_word_lst = ['of', 'Of', 'With', 'with']
        vocab = Vocabulary().add_word_lst(word_lst)
        vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in word_lst+no_create_word_lst]])
        words = embed(words)

        lowered_word_lst = [word.lower() for word in word_lst]
        lowered_no_create_word_lst = [word.lower() for word in no_create_word_lst]
        lowered_vocab = Vocabulary().add_word_lst(lowered_word_lst)
        lowered_vocab.add_word_lst(lowered_no_create_word_lst, no_create_entry=True)
        lowered_embed = StaticEmbedding(lowered_vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=False)
        lowered_words = torch.LongTensor([[lowered_vocab.to_index(word) for word in lowered_word_lst+lowered_no_create_word_lst]])
        lowered_words = lowered_embed(lowered_words)

        all_words = word_lst + no_create_word_lst

        for idx, (word_i, word_j) in enumerate(zip(words[0], lowered_words[0])):
            with self.subTest(idx=idx, word=all_words[idx]):
                assert torch.sum(word_i == word_j).eq(lowered_embed.embed_size)

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_same_vector4(self):
        # 验证在有min_freq下的lower
        word_lst = ["The", "the", "the", "The", "a", "A"]
        no_create_word_lst = ['of', 'Of', "Of", "of", 'With', 'with']
        all_words = word_lst[:-2] + no_create_word_lst[:-2]
        vocab = Vocabulary(min_freq=2).add_word_lst(word_lst)
        vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=True)
        words = torch.LongTensor([[vocab.to_index(word) for word in all_words]])
        words = embed(words)

        lowered_word_lst = [word.lower() for word in word_lst]
        lowered_no_create_word_lst = [word.lower() for word in no_create_word_lst]
        lowered_vocab = Vocabulary().add_word_lst(lowered_word_lst)
        lowered_vocab.add_word_lst(lowered_no_create_word_lst, no_create_entry=True)
        lowered_embed = StaticEmbedding(lowered_vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=False)
        lowered_words = torch.LongTensor([[lowered_vocab.to_index(word.lower()) for word in all_words]])
        lowered_words = lowered_embed(lowered_words)

        for idx in range(len(all_words)):
            word_i, word_j = words[0, idx], lowered_words[0, idx]
            with self.subTest(idx=idx, word=all_words[idx]):
                assert torch.sum(word_i == word_j).eq(lowered_embed.embed_size)

    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_same_vector5(self):
        # 检查通过使用min_freq后的word是否内容一致
        word_lst = ["they", "the", "they", "the", 'he', 'he', "a", "A"]
        no_create_word_lst = ['of', "of", "she", "she", 'With', 'with']
        all_words = word_lst[:-2] + no_create_word_lst[:-2]
        vocab = Vocabulary().add_word_lst(word_lst)
        vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=False, min_freq=2)
        words = torch.LongTensor([[vocab.to_index(word) for word in all_words]])
        words = embed(words)

        min_freq_vocab = Vocabulary(min_freq=2).add_word_lst(word_lst)
        min_freq_vocab.add_word_lst(no_create_word_lst, no_create_entry=True)
        min_freq_embed = StaticEmbedding(min_freq_vocab, model_dir_or_name='en-glove-6B-100d',
                                lower=False)
        min_freq_words = torch.LongTensor([[min_freq_vocab.to_index(word.lower()) for word in all_words]])
        min_freq_words = min_freq_embed(min_freq_words)

        for idx in range(len(all_words)):
            word_i, word_j = words[0, idx], min_freq_words[0, idx]
            with self.subTest(idx=idx, word=all_words[idx]):
                assert torch.sum(word_i == word_j).eq(min_freq_embed.embed_size)
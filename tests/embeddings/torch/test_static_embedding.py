import pytest
import os

from fastNLP.embeddings.torch import StaticEmbedding
from fastNLP import Vocabulary
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
import numpy as np

tests_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

@pytest.mark.torch
class TestLoad:
    def test_norm1(self):
        # 测试只对可以找到的norm
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_norm_found_vector=True)
        assert round(torch.norm(embed(torch.LongTensor([[2]]))).item(), 4) == 1
        assert torch.norm(embed(torch.LongTensor([[4]]))).item() != 1

    def test_norm2(self):
        # 测试对所有都norm
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                normalize=True)
        assert round(torch.norm(embed(torch.LongTensor([[2]]))).item(), 4) == 1
        assert round(torch.norm(embed(torch.LongTensor([[4]]))).item(), 4) == 1

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
                assert embed(torch.LongTensor([vocab.to_index(word)])).tolist()[0] == embed(torch.LongTensor([1])).tolist()[0]

        def check_vector_equal(words, vocab, embed, embed_dict, lower=False):
            for word in words:
                index = vocab.to_index(word)
                v1 = embed(torch.LongTensor([index])).tolist()[0]
                if lower:
                    word = word.lower()
                v2 = embed_dict[word]
                for v1i, v2i in zip(v1, v2):
                    assert np.allclose(v1i, v2i)
        embed_dict = read_static_embed(tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt')

        # 测试是否只使用pretrain的word
        vocab = Vocabulary().add_word_lst(['the', 'a', 'notinfile'])
        vocab.add_word('of', no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True)
        #   notinfile应该被置为unk
        check_vector_equal(['the', 'a', 'of'], vocab, embed, embed_dict)
        check_word_unk(['notinfile'], vocab, embed)

        # 测试在大小写情况下的使用
        vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile'])
        vocab.add_word('Of', no_create_entry=True)
        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True)
        check_word_unk(['The', 'Of', 'notinfile'], vocab, embed)  # 这些词应该找不到
        check_vector_equal(['a'], vocab, embed, embed_dict)

        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True, lower=True)
        check_vector_equal(['The', 'Of', 'a'], vocab, embed, embed_dict, lower=True)
        check_word_unk(['notinfile'], vocab, embed)

        # 测试min_freq
        vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile1', 'A', 'notinfile2', 'notinfile2'])
        vocab.add_word('Of', no_create_entry=True)

        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt',
                                only_use_pretrain_word=True, lower=True, min_freq=2, only_train_min_freq=True)

        check_vector_equal(['Of', 'a'], vocab, embed, embed_dict, lower=True)
        check_word_unk(['notinfile1', 'The', 'notinfile2'], vocab, embed)

    def test_sequential_index(self):
        # 当不存在no_create_entry时，words_to_words应该是顺序的
        vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile1', 'A', 'notinfile2', 'notinfile2'])
        embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt')
        for index,i in enumerate(embed.words_to_words):
            assert index==i

        embed_dict = read_static_embed(tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                         'glove.6B.50d_test.txt')

        for word, index in vocab:
            if word in embed_dict:
                index = vocab.to_index(word)
                v1 = embed(torch.LongTensor([index])).tolist()[0]
                v2 = embed_dict[word]
                for v1i, v2i in zip(v1, v2):
                    assert np.allclose(v1i, v2i)

    def test_save_load_static_embed(self):
        static_test_folder = 'static_save_test'
        try:
            # 测试包含no_create_entry
            os.makedirs(static_test_folder, exist_ok=True)

            vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile1', 'A'])
            vocab.add_word_lst(['notinfile2', 'notinfile2'], no_create_entry=True)
            embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                             'glove.6B.50d_test.txt')
            embed.save(static_test_folder)
            load_embed = StaticEmbedding.load(static_test_folder)
            words = torch.randint(len(vocab), size=(2, 20))
            assert (embed(words) - load_embed(words)).sum() == 0

            # 测试不包含no_create_entry
            vocab = Vocabulary().add_word_lst(['The', 'a', 'notinfile1', 'A'])
            embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                             'glove.6B.50d_test.txt')
            embed.save(static_test_folder)
            load_embed = StaticEmbedding.load(static_test_folder)
            words = torch.randint(len(vocab), size=(2, 20))
            assert (embed(words) - load_embed(words)).sum() == 0

            # 测试lower, min_freq
            vocab = Vocabulary().add_word_lst(['The', 'the', 'the', 'A', 'a', 'B'])
            embed = StaticEmbedding(vocab, model_dir_or_name=tests_folder+'/helpers/data/embedding/small_static_embedding/'
                                                             'glove.6B.50d_test.txt', min_freq=2, lower=True)
            embed.save(static_test_folder)
            load_embed = StaticEmbedding.load(static_test_folder)
            words = torch.randint(len(vocab), size=(2, 20))
            assert (embed(words) - load_embed(words)).sum() == 0

            # 测试random的embedding
            vocab = Vocabulary().add_word_lst(['The', 'the', 'the', 'A', 'a', 'B'])
            vocab = vocab.add_word_lst(['b'], no_create_entry=True)
            embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=4, min_freq=2, lower=True,
                                    normalize=True)
            embed.weight.data += 0.2  # 使得它不是normalize
            embed.save(static_test_folder)
            load_embed = StaticEmbedding.load(static_test_folder)
            words = torch.randint(len(vocab), size=(2, 20))
            assert (embed(words) - load_embed(words)).sum()==0

        finally:
            if os.path.isdir(static_test_folder):
                import shutil
                shutil.rmtree(static_test_folder)


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


@pytest.mark.torch
class TestRandomSameEntry:
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

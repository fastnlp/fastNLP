import pytest
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch

from fastNLP import Vocabulary, DataSet, Instance
from fastNLP.embeddings.torch.char_embedding import LSTMCharEmbedding, CNNCharEmbedding


class TestCharEmbed:
    @pytest.mark.test
    def test_case_1(self):
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['Jack'])])
        vocab = Vocabulary().from_dataset(ds, field_name='words')
        assert len(vocab)==5
        embed = LSTMCharEmbedding(vocab, embed_size=3)
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        assert tuple(y.size()) == (2, 3, 3)

    @pytest.mark.test
    def test_case_2(self):
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['Jack'])])
        vocab = Vocabulary().from_dataset(ds, field_name='words')
        assert len(vocab)==5
        embed = CNNCharEmbedding(vocab, embed_size=3)
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        assert tuple(y.size()) == (2, 3, 3)

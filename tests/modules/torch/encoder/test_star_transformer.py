import pytest

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from fastNLP.modules.torch.encoder.star_transformer import StarTransformer


@pytest.mark.torch
class TestStarTransformer:
    def test_1(self):
        model = StarTransformer(num_layers=6, hidden_size=100, num_head=8, head_dim=20, max_len=100)
        x = torch.rand(16, 45, 100)
        mask = torch.ones(16, 45).byte()
        y, yn = model(x, mask)
        assert (tuple(y.size()) == (16, 45, 100))
        assert (tuple(yn.size()) == (16, 100))

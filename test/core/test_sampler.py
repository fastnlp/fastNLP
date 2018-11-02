import torch

from fastNLP.core.sampler import convert_to_torch_tensor, SequentialSampler, RandomSampler, \
    k_means_1d, k_means_bucketing, simple_sort_bucketing


def test_convert_to_torch_tensor():
    data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 3, 4, 5, 2]]
    ans = convert_to_torch_tensor(data, False)
    assert isinstance(ans, torch.Tensor)
    assert tuple(ans.shape) == (3, 5)


def test_sequential_sampler():
    sampler = SequentialSampler()
    data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    for idx, i in enumerate(sampler(data)):
        assert idx == i


def test_random_sampler():
    sampler = RandomSampler()
    data = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    ans = [data[i] for i in sampler(data)]
    assert len(ans) == len(data)
    for d in ans:
        assert d in data


def test_k_means():
    centroids, assign = k_means_1d([21, 3, 25, 7, 9, 22, 4, 6, 28, 10], 2, max_iter=5)
    centroids, assign = list(centroids), list(assign)
    assert len(centroids) == 2
    assert len(assign) == 10


def test_k_means_bucketing():
    res = k_means_bucketing([21, 3, 25, 7, 9, 22, 4, 6, 28, 10], [None, None])
    assert len(res) == 2


def test_simple_sort_bucketing():
    _ = simple_sort_bucketing([21, 3, 25, 7, 9, 22, 4, 6, 28, 10])
    assert len(_) == 10

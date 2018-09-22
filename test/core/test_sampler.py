import torch

from fastNLP.core.sampler import convert_to_torch_tensor, SequentialSampler, RandomSampler


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


if __name__ == "__main__":
    test_sequential_sampler()

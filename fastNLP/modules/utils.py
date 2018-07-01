import torch


def mask_softmax(matrix, mask):
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=-1)
    else:
        raise NotImplementedError
    return result

import torch


def mask_softmax(matrix, mask):
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=-1)
    else:
        raise NotImplementedError
    return result


def seq_mask(seq_len, max_len):
    mask = [torch.ge(torch.LongTensor(seq_len), i + 1) for i in range(max_len)]
    mask = torch.stack(mask, 1)
    return mask

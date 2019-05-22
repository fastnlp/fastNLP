# Code Modified from https://github.com/carpedm20/ENAS-pytorch

from collections import defaultdict
import collections

import numpy as np
import torch
from torch.autograd import Variable


def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


Node = collections.namedtuple('Node', ['id', 'name'])


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x
    
    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]
    
    return x.item()

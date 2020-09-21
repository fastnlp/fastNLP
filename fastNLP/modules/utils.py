r"""
.. todo::
    doc
"""

__all__ = [
    "initial_parameter",
    "summary"
]

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.init as init


def initial_parameter(net, initial_method=None):
    r"""A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param str initial_method: one of the following initializations.

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_
    
    def weights_init(m):
        # classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):  # for all the cnn
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)  # weight
                else:
                    init.normal_(w.data)  # bias
        elif m is not None and hasattr(m, 'weight') and \
                hasattr(m.weight, "requires_grad"):
                if len(m.weight.size()) > 1:
                    init_method(m.weight.data)
                else:
                    init.normal_(m.weight.data)  # batchnorm or layernorm
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")
    
    net.apply(weights_init)


def summary(model: nn.Module):
    r"""
    得到模型的总参数量

    :params model: Pytorch 模型
    :return tuple: 包含总参数量，可训练参数量，不可训练参数量
    """
    train = []
    nontrain = []
    buffer = []
    
    def layer_summary(module: nn.Module):
        def count_size(sizes):
            return reduce(lambda x, y: x * y, sizes)
        
        for p in module.parameters(recurse=False):
            if p.requires_grad:
                train.append(count_size(p.shape))
            else:
                nontrain.append(count_size(p.shape))
        for p in module.buffers():
            buffer.append(count_size(p.shape))
        for subm in module.children():
            layer_summary(subm)
    
    layer_summary(model)
    total_train = sum(train)
    total_nontrain = sum(nontrain)
    total = total_train + total_nontrain
    strings = []
    strings.append('Total params: {:,}'.format(total))
    strings.append('Trainable params: {:,}'.format(total_train))
    strings.append('Non-trainable params: {:,}'.format(total_nontrain))
    strings.append("Buffer params: {:,}".format(sum(buffer)))
    max_len = len(max(strings, key=len))
    bar = '-' * (max_len + 3)
    strings = [bar] + strings + [bar]
    print('\n'.join(strings))
    return total, total_train, total_nontrain


def get_dropout_mask(drop_p: float, tensor: torch.Tensor):
    r"""
    根据tensor的形状，生成一个mask

    :param drop_p: float, 以多大的概率置为0。
    :param tensor: torch.Tensor
    :return: torch.FloatTensor. 与tensor一样的shape
    """
    mask_x = torch.ones_like(tensor)
    nn.functional.dropout(mask_x, p=drop_p,
                          training=False, inplace=True)
    return mask_x

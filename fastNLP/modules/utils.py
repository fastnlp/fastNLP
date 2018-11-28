import torch
import torch.nn as nn
import torch.nn.init as init


def mask_softmax(matrix, mask):
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=-1)
    else:
        raise NotImplementedError
    return result


def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param initial_method: str, one of the following initializations

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
        elif hasattr(m, 'weight') and m.weight.requires_grad:
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)  # weight
                    else:
                        init.normal_(w.data)  # bias
                # print("init else")

    net.apply(weights_init)


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    :param seq_len: list or torch.Tensor, the lengths of sequences in a batch.
    :param max_len: int, the maximum sequence length in a batch.
    :return mask: torch.LongTensor, [batch_size, max_len]

    """
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.LongTensor(seq_len)
    seq_len = seq_len.view(-1, 1).long()   # [batch_size, 1]
    seq_range = torch.arange(start=0, end=max_len, dtype=torch.long, device=seq_len.device).view(1, -1) # [1, max_len]
    return torch.gt(seq_len, seq_range) # [batch_size, max_len]

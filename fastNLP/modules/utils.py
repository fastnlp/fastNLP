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


"""
    Codes from FudanParser
"""
from collections import defaultdict

import numpy as np
import torch


def expand_gt(gt):
    """expand_gt: Expand ground truth to matrix
    Arguments:
        gt: tensor of (n, l)
    Return:
        f: ground truth matrix of (n, l), $gt[i][j] = k$ leads to $f[i][j][k] = 1$.
    """
    n, l = gt.shape
    ret = torch.zeros(n, l, l).long()
    for i in range(n):
        ret[i][torch.arange(l).long(), gt[i]] = 1
    return ret


def greedy_decoding(arc_f):
    """greedy_decoding
    Arguments:
        arc_f: a tensor in shape of (n, l+1, l+1)
               length of the sentence is l and index 0 is <root>
    Output:
        arc_pred: a tensor in shape of (n, l), indicating the head words
    """

    f_arc = arc_f[:, 1:, :]  # ignore the root
    _, arc_pred = torch.max(f_arc.data, dim=-1, keepdim=False)
    return arc_pred


def mst_decoding(arc_f):
    batch_size = arc_f.shape[0]
    length = arc_f.shape[1]
    arc_score = arc_f.data.cpu()
    pred_collection = []
    for i in range(batch_size):
        head = mst(arc_score[i].numpy())
        pred_collection.append(head[1:].reshape((1, length - 1)))
    arc_pred = torch.LongTensor(np.concatenate(pred_collection, axis=0)).type_as(arc_f).long()
    return arc_pred


def outer_product(features):
    """InterProduct: Get inter sequence product of features
    Arguments:
        features: feature vectors of sequence in the shape of (n, l, h)
    Return:
        f: product result in (n, l, l, h) shape
    """
    n, l, c = features.shape
    features = features.contiguous()
    x = features.view(n, l, 1, c)
    x = x.expand(n, l, l, c)
    y = features.view(n, 1, l, c).contiguous()
    y = y.expand(n, l, l, c)
    return x * y


def outer_concat(features):
    """InterProduct: Get inter sequence concatenation of features
    Arguments:
        features: feature vectors of sequence in the shape of (n, l, h)
    Return:
        f: product result in (n, l, l, h) shape
    """
    n, l, c = features.shape
    x = features.contiguous().view(n, l, 1, c)
    x = x.expand(n, l, l, c)
    y = features.view(n, 1, l, c)
    y = y.expand(n, l, l, c)
    return torch.cat((x, y), dim=3)


def mst(scores):
    """
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692  # NOQA
    """
    length = scores.shape[0]
    min_score = scores.min() - 1
    eye = np.eye(length)
    scores = scores * (1 - eye) + min_score * eye
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)
    vertices = set((0,))
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = min_score
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm  # NOQA
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py  # NOQA
    """
    _index = 0
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []

    def _strongconnect(v):
        nonlocal _index
        _indices[v] = _index
        _lowlinks[v] = _index
        _index += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()
            while True:
                w = _stack.pop()
                _onstack[w] = False
                SCC.add(w)
                if not (w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


# https://github.com/alykhantejani/nninit/blob/master/nninit.py
def orthogonal(tensor, gain=1):
    """Fills the input Tensor or Variable with a (semi) orthogonal matrix. The input tensor must have at least 2 dimensions,
       and for tensors with more than 2 dimensions the trailing dimensions are flattened. viewed as 2D representation with
       rows equal to the first dimension and columns equal to the product of  as a sparse matrix, where the non-zero elements
       will be drawn from a normal distribution with mean=0 and std=`std`.
       Reference: "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" - Saxe, A. et al.
    Args:
        tensor: a n-dimension torch.Tensor, where n >= 2
        gain: optional gain to be applied
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.orthogonal(w)
    """
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported.")

    flattened_shape = (tensor.size(0), int(np.prod(tensor.detach().numpy().shape[1:])))
    flattened = torch.Tensor(flattened_shape[0], flattened_shape[1]).normal_(0, 1)

    u, s, v = np.linalg.svd(flattened.numpy(), full_matrices=False)
    if u.shape == flattened.detach().numpy().shape:
        tensor.view_as(flattened).copy_(torch.from_numpy(u))
    else:
        tensor.view_as(flattened).copy_(torch.from_numpy(v))

    tensor.mul_(gain)
    with torch.no_grad():
        return tensor


def generate_step_dropout(masks, hidden_dim, step_dropout, training=False):
    # assume batch first
    # import pdb
    # pdb.set_trace()

    batch, length = masks.size()
    if not training:
        return torch.ones(batch, length, hidden_dim).fill_(1 - step_dropout).cuda(masks.device) * masks.view(batch,
                                                                                                             length, 1)
    masked = torch.zeros(batch, 1, hidden_dim).fill_(step_dropout)
    masked = torch.bernoulli(masked).repeat(1, length, 1)
    masked = masked.cuda(masks.device) * masks.view(batch, length, 1)
    return masked

r"""undocumented"""

__all__ = [
    "viterbi_decode"
]
import torch


def viterbi_decode(logits, transitions, mask=None, unpad=False):
    r"""
    给定一个特征矩阵以及转移分数矩阵，计算出最佳的路径以及对应的分数

    :param torch.FloatTensor logits: batch_size x max_len x num_tags，特征矩阵。
    :param torch.FloatTensor transitions:  n_tags x n_tags，[i, j]位置的值认为是从tag i到tag j的转换; 或者(n_tags+2) x
        (n_tags+2), 其中n_tag是start的index, n_tags+1是end的index; 如果要i->j之间不允许越迁，就把transitions中(i,j)设置为很小的
        负数，例如-10000000.0
    :param torch.ByteTensor mask: batch_size x max_len, 为0的位置认为是pad；如果为None，则认为没有padding。
    :param bool unpad: 是否将结果删去padding。False, 返回的是batch_size x max_len的tensor; True，返回的是
        List[List[int]], 内部的List[int]为每个sequence的label，已经除去pad部分，即每个List[int]的长度是这
        个sample的有效长度。
    :return: 返回 (paths, scores)。
                paths: 是解码后的路径, 其值参照unpad参数.
                scores: torch.FloatTensor, size为(batch_size,), 对应每个最优路径的分数。

    """
    batch_size, seq_len, n_tags = logits.size()
    if transitions.size(0) == n_tags+2:
        include_start_end_trans = True
    elif transitions.size(0) == n_tags:
        include_start_end_trans = False
    else:
        raise RuntimeError("The shapes of transitions and feats are not " \
            "compatible.")
    logits = logits.transpose(0, 1).data  # L, B, H
    if mask is not None:
        mask = mask.transpose(0, 1).data.eq(True)  # L, B
    else:
        mask = logits.new_ones((seq_len, batch_size), dtype=torch.uint8).eq(1)

    trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data

    # dp
    vpath = logits.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)
    vscore = logits[0]
    if include_start_end_trans:
        vscore += transitions[n_tags, :n_tags]

    for i in range(1, seq_len):
        prev_score = vscore.view(batch_size, n_tags, 1)
        cur_score = logits[i].view(batch_size, 1, n_tags)
        score = prev_score + trans_score + cur_score
        best_score, best_dst = score.max(1)
        vpath[i] = best_dst
        vscore = best_score.masked_fill(mask[i].eq(False).view(batch_size, 1), 0) + \
                 vscore.masked_fill(mask[i].view(batch_size, 1), 0)

    if include_start_end_trans:
        vscore += transitions[:n_tags, n_tags + 1].view(1, -1)
    # backtrace
    batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
    seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)
    lens = (mask.long().sum(0) - 1)
    # idxes [L, B], batched idx from seq_len-1 to 0
    idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq_len

    ans = logits.new_empty((seq_len, batch_size), dtype=torch.long)
    ans_score, last_tags = vscore.max(1)
    ans[idxes[0], batch_idx] = last_tags
    for i in range(seq_len - 1):
        last_tags = vpath[idxes[i], batch_idx, last_tags]
        ans[idxes[i + 1], batch_idx] = last_tags
    ans = ans.transpose(0, 1)
    if unpad:
        paths = []
        for idx, seq_len in enumerate(lens):
            paths.append(ans[idx, :seq_len + 1].tolist())
    else:
        paths = ans
    return paths, ans_score

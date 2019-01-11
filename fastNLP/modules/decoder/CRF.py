import torch
from torch import nn

from fastNLP.modules.utils import initial_parameter


def log_sum_exp(x, dim=-1):
    max_value, _ = x.max(dim=dim, keepdim=True)
    res = torch.log(torch.sum(torch.exp(x - max_value), dim=dim, keepdim=True)) + max_value
    return res.squeeze(dim)


def seq_len_to_byte_mask(seq_lens):
    # usually seq_lens: LongTensor, batch_size
    # return value: ByteTensor, batch_size x max_len
    batch_size = seq_lens.size(0)
    max_len = seq_lens.max()
    broadcast_arange = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(seq_lens.device)
    mask = broadcast_arange.float().lt(seq_lens.float().view(-1, 1))
    return mask

def allowed_transitions(id2label, encoding_type='bio'):
    """

    :param id2label: dict, key是label的indices，value是str类型的tag或tag-label。value可以是只有tag的, 比如"B", "M"; 也可以是
        "B-NN", "M-NN", tag和label之间一定要用"-"隔开。一般可以通过Vocabulary.get_id2word()id2label。
    :param encoding_type: str, 支持"bio", "bmes"。
    :return:List[Tuple(int, int)]], 内部的Tuple是(from_tag_id, to_tag_id)。 返回的结果考虑了start和end，比如"BIO"中，B、O可以
        位于序列的开端，而I不行。所以返回的结果中会包含(start_idx, B_idx), (start_idx, O_idx), 但是不包含(start_idx, I_idx).
        start_idx=len(id2label), end_idx=len(id2label)+1。
    """
    num_tags = len(id2label)
    start_idx = num_tags
    end_idx = num_tags + 1
    encoding_type = encoding_type.lower()
    allowed_trans = []
    id_label_lst = list(id2label.items()) + [(start_idx, 'start'), (end_idx, 'end')]
    def split_tag_label(from_label):
        from_label = from_label.lower()
        if from_label in ['start', 'end']:
            from_tag = from_label
            from_label = ''
        else:
            from_tag = from_label[:1]
            from_label = from_label[2:]
        return from_tag, from_label

    for from_id, from_label in id_label_lst:
        if from_label in ['<pad>', '<unk>']:
            continue
        from_tag, from_label = split_tag_label(from_label)
        for to_id, to_label in id_label_lst:
            if to_label in ['<pad>', '<unk>']:
                continue
            to_tag, to_label = split_tag_label(to_label)
            if is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
                allowed_trans.append((from_id, to_id))
    return allowed_trans

def is_transition_allowed(encoding_type, from_tag, from_label, to_tag, to_label):
    """

    :param encoding_type: str, 支持"BIO", "BMES"。
    :param from_tag: str, 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param from_label: str, 比如"PER", "LOC"等label
    :param to_tag: str, 比如"B", "M"之类的标注tag. 还包括start, end等两种特殊tag
    :param to_label: str, 比如"PER", "LOC"等label
    :return: bool，能否跃迁
    """
    if to_tag=='start' or from_tag=='end':
        return False
    encoding_type = encoding_type.lower()
    if encoding_type == 'bio':
        """
        第一行是to_tag, 第一列是from_tag. y任意条件下可转，-只有在label相同时可转，n不可转
        +-------+---+---+---+-------+-----+
        |       | B | I | O | start | end |
        +-------+---+---+---+-------+-----+
        |   B   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   I   | y | - | y | n     | y   |
        +-------+---+---+---+-------+-----+
        |   O   | y | n | y | n     | y   |
        +-------+---+---+---+-------+-----+
        | start | y | n | y | n     | n   |
        +-------+---+---+---+-------+-----+
        | end   | n | n | n | n     | n   |
        +-------+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ('b', 'o')
        elif from_tag in ['b', 'i']:
            return any([to_tag in ['end', 'b', 'o'], to_tag=='i' and from_label==to_label])
        elif from_tag == 'o':
            return to_tag in ['end', 'b', 'o']
        else:
            raise ValueError("Unexpect tag {}. Expect only 'B', 'I', 'O'.".format(from_tag))

    elif encoding_type == 'bmes':
        """
        第一行是to_tag, 第一列是from_tag，y任意条件下可转，-只有在label相同时可转，n不可转
        +-------+---+---+---+---+-------+-----+
        |       | B | M | E | S | start | end |
        +-------+---+---+---+---+-------+-----+
        |   B   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   M   | n | - | - | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |   E   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        |   S   | y | n | n | y |   n   |  y  |
        +-------+---+---+---+---+-------+-----+
        | start | y | n | n | y |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        |  end  | n | n | n | n |   n   |  n  |
        +-------+---+---+---+---+-------+-----+
        """
        if from_tag == 'start':
            return to_tag in ['b', 's']
        elif from_tag == 'b':
            return to_tag in ['m', 'e'] and from_label==to_label
        elif from_tag == 'm':
            return to_tag in ['m', 'e'] and from_label==to_label
        elif from_tag in ['e', 's']:
            return to_tag in ['b', 's', 'end']
        else:
            raise ValueError("Unexpect tag type {}. Expect only 'B', 'M', 'E', 'S'.".format(from_tag))
    else:
        raise ValueError("Only support BIO, BMES encoding type, got {}.".format(encoding_type))


class ConditionalRandomField(nn.Module):
    def __init__(self, num_tags, include_start_end_trans=False, allowed_transitions=None, initial_method=None):
        """

        :param num_tags: int, 标签的数量。
        :param include_start_end_trans: bool, 是否包含起始tag
        :param allowed_transitions: List[Tuple[from_tag_id(int), to_tag_id(int)]]. 允许的跃迁，可以通过allowed_transitions()得到。
            如果为None，则所有跃迁均为合法
        :param initial_method:
        """

        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.num_tags = num_tags

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.trans_m = nn.Parameter(torch.randn(num_tags, num_tags))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(num_tags))
            self.end_scores = nn.Parameter(torch.randn(num_tags))

        if allowed_transitions is None:
            constrain = torch.zeros(num_tags + 2, num_tags + 2)
        else:
            constrain = torch.ones(num_tags + 2, num_tags + 2) * -1000
            for from_tag_id, to_tag_id in allowed_transitions:
                constrain[from_tag_id, to_tag_id] = 0
        self._constrain = nn.Parameter(constrain, requires_grad=False)

        # self.reset_parameter()
        initial_parameter(self, initial_method)
    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_m)
        if self.include_start_end_trans:
            nn.init.normal_(self.start_scores)
            nn.init.normal_(self.end_scores)

    def _normalizer_likelihood(self, logits, mask):
        """Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.

        :param logits:FloatTensor, max_len x batch_size x num_tags
        :param mask:ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha += self.start_scores.view(1, -1)

        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = log_sum_exp(tmp, 1) * mask[i].view(batch_size, 1) + alpha * (1 - mask[i]).view(batch_size, 1)

        if self.include_start_end_trans:
            alpha += self.end_scores.view(1, -1)

        return log_sum_exp(alpha, 1)

    def _glod_score(self, logits, tags, mask):
        """
        Compute the score for the gold path.
        :param logits: FloatTensor, max_len x batch_size x num_tags
        :param tags: LongTensor, max_len x batch_size
        :param mask: ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)

        # trans_socre [L-1, B]
        trans_score = self.trans_m[tags[:seq_len-1], tags[1:]] * mask[1:, :]
        # emit_score [L, B]
        emit_score = logits[seq_idx.view(-1,1), batch_idx.view(1,-1), tags] * mask
        # score [L-1, B]
        score = trans_score + emit_score[:seq_len-1, :]
        score = score.sum(0) + emit_score[-1] * mask[-1]
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[batch_idx, tags[last_idx, batch_idx]]
            score += st_scores + ed_scores
        # return [B,]
        return score

    def forward(self, feats, tags, mask):
        """
        Calculate the neg log likelihood
        :param feats:FloatTensor, batch_size x max_len x num_tags
        :param tags:LongTensor, batch_size x max_len
        :param mask:ByteTensor batch_size x max_len
        :return:FloatTensor, batch_size
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._glod_score(feats, tags, mask)

        return all_path_score - gold_path_score

    def viterbi_decode(self, data, mask, get_score=False, unpad=False):
        """
        Given a feats matrix, return best decode path and best score.
        :param data:FloatTensor, batch_size x max_len x num_tags
        :param mask:ByteTensor batch_size x max_len
        :param get_score: bool, whether to output the decode score.
        :param unpad: bool, 是否将结果unpad,
                            如果False, 返回的是batch_size x max_len的tensor，
                            如果True，返回的是List[List[int]], List[int]为每个sequence的label，已经unpadding了，即每个
                                List[int]的长度是这个sample的有效长度
        :return: 如果get_score为False，返回结果根据unpadding变动
                 如果get_score为True, 返回 (paths, List[float], )。第一个仍然是解码后的路径(根据unpad变化)，第二个List[Float]
                    为每个seqence的解码分数。

        """
        batch_size, seq_len, n_tags = data.size()
        data = data.transpose(0, 1).data # L, B, H
        mask = mask.transpose(0, 1).data.float() # L, B

        # dp
        vpath = data.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)
        vscore = data[0]
        transitions = self._constrain.data.clone()
        transitions[:n_tags, :n_tags] += self.trans_m.data
        if self.include_start_end_trans:
            transitions[n_tags, :n_tags] += self.start_scores.data
            transitions[:n_tags, n_tags+1] += self.end_scores.data

        vscore += transitions[n_tags, :n_tags]
        trans_score = transitions[:n_tags, :n_tags].view(1, n_tags, n_tags).data
        for i in range(1, seq_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = data[i].view(batch_size, 1, n_tags)
            score = prev_score + trans_score + cur_score
            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            vscore = best_score * mask[i].view(batch_size, 1) + vscore * (1 - mask[i]).view(batch_size, 1)

        vscore += transitions[:n_tags, n_tags+1].view(1, -1)

        # backtrace
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=data.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=data.device)
        lens = (mask.long().sum(0) - 1)
        # idxes [L, B], batched idx from seq_len-1 to 0
        idxes = (lens.view(1,-1) - seq_idx.view(-1,1)) % seq_len

        ans = data.new_empty((seq_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(seq_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i+1], batch_idx] = last_tags
        ans = ans.transpose(0, 1)
        if unpad:
            paths = []
            for idx, seq_len in enumerate(lens):
                paths.append(ans[idx, :seq_len+1].tolist())
        else:
            paths = ans
        if get_score:
            return paths, ans_score.tolist()
        return paths

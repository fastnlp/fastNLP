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
    broadcast_arange = torch.arange(max_len).view(1, -1).repeat(batch_size, 1)
    mask = broadcast_arange.lt(seq_lens.float().view(-1, 1))
    return mask


class ConditionalRandomField(nn.Module):
    def __init__(self, tag_size, include_start_end_trans=True ,initial_method = None):
        """
        :param tag_size: int, num of tags
        :param include_start_end_trans: bool, whether to include start/end tag
        """
        super(ConditionalRandomField, self).__init__()

        self.include_start_end_trans = include_start_end_trans
        self.tag_size = tag_size

        # the meaning of entry in this matrix is (from_tag_id, to_tag_id) score
        self.transition_m = nn.Parameter(torch.randn(tag_size, tag_size))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(tag_size))
            self.end_scores = nn.Parameter(torch.randn(tag_size))

        # self.reset_parameter()
        initial_parameter(self, initial_method)
    def reset_parameter(self):
        nn.init.xavier_normal_(self.transition_m)
        if self.include_start_end_trans:
            nn.init.normal_(self.start_scores)
            nn.init.normal_(self.end_scores)

    def _normalizer_likelihood(self, feats, masks):
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        :param feats:FloatTensor, batch_size x max_len x tag_size
        :param masks:ByteTensor, batch_size x max_len
        :return:FloatTensor, batch_size
        """
        batch_size, max_len, _ = feats.size()

        # alpha, batch_size x tag_size
        if self.include_start_end_trans:
            alpha = self.start_scores.view(1, -1) + feats[:, 0]
        else:
            alpha = feats[:, 0]

        # broadcast_trans_m, the meaning of entry in this matrix is [batch_idx, to_tag_id, from_tag_id]
        broadcast_trans_m = self.transition_m.permute(
            1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        # loop
        for i in range(1, max_len):
            emit_score = feats[:, i].unsqueeze(2)
            new_alpha = broadcast_trans_m + alpha.unsqueeze(1) + emit_score

            new_alpha = log_sum_exp(new_alpha, dim=2)

            alpha = new_alpha * \
                    masks[:, i:i + 1].float() + alpha * \
                    (1 - masks[:, i:i + 1].float())

        if self.include_start_end_trans:
            alpha = alpha + self.end_scores.view(1, -1)

        return log_sum_exp(alpha)

    def _glod_score(self, feats, tags, masks):
        """
        Compute the score for the gold path.
        :param feats: FloatTensor, batch_size x max_len x tag_size
        :param tags: LongTensor, batch_size x max_len
        :param masks: ByteTensor, batch_size x max_len
        :return:FloatTensor, batch_size
        """
        batch_size, max_len, _ = feats.size()

        # alpha, B x 1
        if self.include_start_end_trans:
            alpha = self.start_scores.view(1, -1).repeat(batch_size, 1).gather(dim=1, index=tags[:, :1]) + \
                    feats[:, 0].gather(dim=1, index=tags[:, :1])
        else:
            alpha = feats[:, 0].gather(dim=1, index=tags[:, :1])

        for i in range(1, max_len):
            trans_score = self.transition_m[(
                tags[:, i - 1], tags[:, i])].unsqueeze(1)
            emit_score = feats[:, i].gather(dim=1, index=tags[:, i:i + 1])
            new_alpha = alpha + trans_score + emit_score

            alpha = new_alpha * \
                    masks[:, i:i + 1].float() + alpha * \
                    (1 - masks[:, i:i + 1].float())

        if self.include_start_end_trans:
            last_tag_index = masks.cumsum(dim=1, dtype=torch.long)[:, -1:] - 1
            last_from_tag_id = tags.gather(dim=1, index=last_tag_index)
            trans_score = self.end_scores.view(
                1, -1).repeat(batch_size, 1).gather(dim=1, index=last_from_tag_id)
            alpha = alpha + trans_score

        return alpha.squeeze(1)

    def forward(self, feats, tags, masks):
        """
        Calculate the neg log likelihood
        :param feats:FloatTensor, batch_size x max_len x tag_size
        :param tags:LongTensor, batch_size x max_len
        :param masks:ByteTensor batch_size x max_len
        :return:FloatTensor, batch_size
        """
        all_path_score = self._normalizer_likelihood(feats, masks)
        gold_path_score = self._glod_score(feats, tags, masks)

        return all_path_score - gold_path_score

    def viterbi_decode(self, feats, masks, get_score=False):
        """
        Given a feats matrix, return best decode path and best score.
        :param feats:
        :param masks:
        :param get_score: bool, whether to output the decode score.
        :return:List[Tuple(List, float)],
        """
        batch_size, max_len, tag_size = feats.size()

        paths = torch.zeros(batch_size, max_len - 1, self.tag_size)
        if self.include_start_end_trans:
            alpha = self.start_scores.repeat(batch_size, 1) + feats[:, 0]
        else:
            alpha = feats[:, 0]
        for i in range(1, max_len):
            new_alpha = alpha.clone()
            for t in range(self.tag_size):
                pre_scores = self.transition_m[:, t].view(
                    1, self.tag_size) + alpha
                max_score, indices = pre_scores.max(dim=1)
                new_alpha[:, t] = max_score + feats[:, i, t]
                paths[:, i - 1, t] = indices
            alpha = new_alpha * masks[:, i:i + 1].float() + alpha * (1 - masks[:, i:i + 1].float())

        if self.include_start_end_trans:
            alpha += self.end_scores.view(1, -1)

        max_scores, indices = alpha.max(dim=1)
        indices = indices.cpu().numpy()
        final_paths = []
        paths = paths.cpu().numpy().astype(int)

        seq_lens = masks.cumsum(dim=1, dtype=torch.long)[:, -1]

        for b in range(batch_size):
            path = [indices[b]]
            for i in range(seq_lens[b] - 2, -1, -1):
                index = paths[b, i, path[-1]]
                path.append(index)
            final_paths.append(path[::-1])
        if get_score:
            return list(zip(final_paths, max_scores.detach().cpu().numpy()))
        else:
            return final_paths

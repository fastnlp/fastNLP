__all__ = [
    "SequenceGenerator"
]
import torch
from .seq2seq_decoder import Decoder
import torch.nn.functional as F
from ...core.utils import _get_model_device
from functools import partial


class SequenceGenerator:
    def __init__(self, decoder: Decoder, max_length=20, num_beams=1,
                 do_sample=True, temperature=1.0, top_k=50, top_p=1.0, bos_token_id=None, eos_token_id=None,
                 repetition_penalty=1, length_penalty=1.0):
        if do_sample:
            self.generate_func = partial(sample_generate, decoder=decoder, max_length=max_length, num_beams=num_beams,
                                         temperature=temperature, top_k=top_k, top_p=top_p, bos_token_id=bos_token_id,
                                         eos_token_id=eos_token_id, repetition_penalty=repetition_penalty,
                                         length_penalty=length_penalty)
        else:
            self.generate_func = partial(greedy_generate, decoder=decoder, max_length=max_length, num_beams=num_beams,
                                         bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                                         repetition_penalty=repetition_penalty,
                                         length_penalty=length_penalty)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.decoder = decoder

    @torch.no_grad()
    def generate(self, tokens=None, past=None):
        """

        :param torch.LongTensor tokens: batch_size x length, 开始的token
        :param past:
        :return:
        """
        # TODO 需要查看如果tokens长度不是1，decode的时候是否还能够直接decode？
        return self.generate_func(tokens=tokens, past=past)


@torch.no_grad()
def greedy_generate(decoder, tokens=None, past=None, max_length=20, num_beams=1,
                    bos_token_id=None, eos_token_id=None,
                    repetition_penalty=1, length_penalty=1.0):
    """
    贪婪地搜索句子

    :param Decoder decoder: Decoder对象
    :param torch.LongTensor tokens: batch_size x len, decode的输入值，如果为None，则自动从bos_token_id开始生成
    :param Past past: 应该包好encoder的一些输出。
    :param int max_length: 生成句子的最大长度。
    :param int num_beams: 使用多大的beam进行解码。
    :param int bos_token_id: 如果tokens传入为None，则使用bos_token_id开始往后解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens, past, max_length, temperature=1, top_k=50, top_p=1,
                                             bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False,
                                             repetition_penalty=repetition_penalty, length_penalty=length_penalty)
    else:
        token_ids = _beam_search_generate(decoder, tokens, past, max_length, num_beams=num_beams,
                                          temperature=1, top_k=50, top_p=1,
                                          bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False,
                                          repetition_penalty=repetition_penalty, length_penalty=length_penalty)

    return token_ids


@torch.no_grad()
def sample_generate(decoder, tokens=None, past=None, max_length=20, num_beams=1, temperature=1.0, top_k=50,
                    top_p=1.0, bos_token_id=None, eos_token_id=None, repetition_penalty=1.0, length_penalty=1.0):
    """
    使用采样的方法生成句子

    :param Decoder decoder: Decoder对象
    :param torch.LongTensor tokens: batch_size x len, decode的输入值，如果为None，则自动从bos_token_id开始生成
    :param Past past: 应该包好encoder的一些输出。
    :param int max_length: 生成句子的最大长度。
    :param int num_beam: 使用多大的beam进行解码。
    :param float temperature: 采样时的退火大小
    :param int top_k: 只在top_k的sample里面采样
    :param float top_p: 介于0,1的值。
    :param int bos_token_id: 如果tokens传入为None，则使用bos_token_id开始往后解码。
    :param int eos_token_id: 结束的token，如果为None，则一定会解码到max_length这么长。
    :param float repetition_penalty: 对重复出现的token多大的惩罚。
    :param float length_penalty: 对每个token（除了eos）按照长度进行一定的惩罚。
    :return:
    """
    # 每个位置在生成的时候会sample生成
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens, past, max_length, temperature=temperature,
                                             top_k=top_k, top_p=top_p,
                                             bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=True,
                                             repetition_penalty=repetition_penalty, length_penalty=length_penalty)
    else:
        token_ids = _beam_search_generate(decoder, tokens, past, max_length, num_beams=num_beams,
                                          temperature=temperature, top_k=top_k, top_p=top_p,
                                          bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=True,
                                          repetition_penalty=repetition_penalty, length_penalty=length_penalty)
    return token_ids


def _no_beam_search_generate(decoder: Decoder, tokens=None, past=None, max_length=20, temperature=1.0, top_k=50,
                             top_p=1.0, bos_token_id=None, eos_token_id=None, do_sample=True,
                             repetition_penalty=1.0, length_penalty=1.0):
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        if past is None:
            raise RuntimeError("You have to specify either `past` or `tokens`.")
        batch_size = past.num_samples()
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `past`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if past is not None:
        assert past.num_samples() == batch_size, "The number of samples in `tokens` and `past` should match."

    if eos_token_id is None:
        _eos_token_id = float('nan')
    else:
        _eos_token_id = eos_token_id

    for i in range(tokens.size(1)):
        scores, past = decoder.decode(tokens[:, :i + 1], past)  # batch_size x vocab_size, Past

    token_ids = tokens.clone()
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1)
    # tokens = tokens[:, -1:]

    while cur_len < max_length:
        scores, past = decoder.decode(tokens, past)  # batch_size x vocab_size, Past

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty  # batch_size x vocab_size
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores)  # 也即除了eos，其他词的分数经过了放大/缩小

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=2)
            probs = F.softmax(scores, dim=-1)

            # 保证至少有一个不是eos的值
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # batch_size
        else:
            next_tokens = torch.argmax(scores, dim=-1)  # batch_size

        next_tokens = next_tokens.masked_fill(dones, 0)  # 对已经搜索完成的sample做padding
        tokens = next_tokens.unsqueeze(1)

        token_ids = torch.cat([token_ids, tokens], dim=-1)  # batch_size x max_len

        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1

        if dones.min() == 1:
            break

    if eos_token_id is not None:
        if cur_len == max_length:
            token_ids[:, -1].masked_fill_(~dones, eos_token_id)  # 若到最长长度仍未到EOS，则强制将最后一个词替换成eos

    return token_ids


def _beam_search_generate(decoder: Decoder, tokens=None, past=None, max_length=20, num_beams=4, temperature=1.0,
                          top_k=50,
                          top_p=1.0, bos_token_id=None, eos_token_id=None, do_sample=True,
                          repetition_penalty=1.0, length_penalty=None) -> torch.LongTensor:
    # 进行beam search
    device = _get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError("You have to specify either `tokens` or `bos_token_id`.")
        if past is None:
            raise RuntimeError("You have to specify either `past` or `tokens`.")
        batch_size = past.num_samples()
        if batch_size is None:
            raise RuntimeError("Cannot infer the number of samples from `past`.")
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long).to(device)
    batch_size = tokens.size(0)
    if past is not None:
        assert past.num_samples() == batch_size, "The number of samples in `tokens` and `past` should match."

    for i in range(tokens.size(1) - 1):  # 如果输入的长度较长，先decode
        scores, past = decoder.decode(tokens[:, :i + 1],
                                      past)  # (batch_size, vocab_size), Past
    scores, past = decoder.decode(tokens, past)  # 这里要传入的是整个句子的长度
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, "num_beams should be smaller than the number of vocabulary size."

    if do_sample:
        probs = F.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=num_beams)  # (batch_size, num_beams)
        logits = probs.log()
        next_scores = logits.gather(dim=1, index=next_tokens)  # (batch_size, num_beams)
    else:
        scores = F.log_softmax(scores, dim=-1)  # (batch_size, vocab_size)
        # 得到(batch_size, num_beams), (batch_size, num_beams)
        next_scores, next_tokens = torch.topk(scores, num_beams, dim=1, largest=True, sorted=True)

    indices = torch.arange(batch_size, dtype=torch.long).to(device)
    indices = indices.repeat_interleave(num_beams)
    decoder.reorder_past(indices, past)

    tokens = tokens.index_select(dim=0, index=indices)  # batch_size * num_beams x length
    # 记录生成好的token (batch_size', cur_len)
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size
    tokens = next_tokens.view(-1, 1)

    beam_scores = next_scores.view(-1)  # batch_size * num_beams

    #  用来记录已经生成好的token的长度
    cur_len = token_ids.size(1)

    hypos = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]
    # 0,num_beams, 2*num_beams, ...
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1).to(token_ids)

    while cur_len < max_length:
        scores, past = decoder.decode(tokens, past)  # batch_size * num_beams x vocab_size, Past

        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)

        if do_sample:
            if temperature > 0 and temperature != 1:
                scores = scores / temperature

            # 多召回一个防止eos
            scores = top_k_top_p_filtering(scores, top_k, top_p, min_tokens_to_keep=num_beams + 1)
            probs = F.softmax(scores, dim=-1)

            # 保证至少有一个不是eos的值
            _tokens = torch.multinomial(probs, num_samples=num_beams + 1)  # batch_size' x (num_beams+1)

            logits = probs.log()
            # 防止全是这个beam的被选中了，且需要考虑eos被选择的情况
            _scores = logits.gather(dim=1, index=_tokens)  # batch_size' x (num_beams+1)
            _scores = _scores + beam_scores[:, None]  # batch_size' x (num_beams+1)
            # 从这里面再选择top的2*num_beam个
            _scores = _scores.view(batch_size, num_beams * (num_beams + 1))
            next_scores, ids = _scores.topk(2 * num_beams, dim=1, largest=True, sorted=True)
            _tokens = _tokens.view(batch_size, num_beams * (num_beams + 1))
            next_tokens = _tokens.gather(dim=1, index=ids)  # (batch_size, 2*num_beams)
            from_which_beam = ids // (num_beams + 1)  # (batch_size, 2*num_beams)
        else:
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            _scores = scores + beam_scores[:, None]  # (batch_size * num_beams, vocab_size)
            _scores = _scores.view(batch_size, -1)  # (batch_size, num_beams*vocab_size)
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            from_which_beam = ids // vocab_size  # (batch_size, 2*num_beams)
            next_tokens = ids % vocab_size  # (batch_size, 2*num_beams)

        #  接下来需要组装下一个batch的结果。
        #  需要选定哪些留下来
        next_scores, sorted_inds = next_scores.sort(dim=-1, descending=True)
        next_tokens = next_tokens.gather(dim=1, index=sorted_inds)
        from_which_beam = from_which_beam.gather(dim=1, index=sorted_inds)

        not_eos_mask = next_tokens.ne(eos_token_id)  # 为1的地方不是eos
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)  # 为1的地方需要保留
        keep_mask = not_eos_mask.__and__(keep_mask)  # 为1的地方是需要进行下一步search的

        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)  # 上面的token是来自哪个beam
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)

        # 更改past状态, 重组token_ids
        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)  # flatten成一维
        decoder.reorder_past(reorder_inds, past)

        flag = True
        if cur_len + 1 == max_length:
            eos_batch_idx = torch.arange(batch_size).to(next_tokens).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).to(token_ids).repeat(batch_size)  # 表示的是indice
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)  # 表示的是从哪个beam获取得到的
        else:
            # 将每个batch中在num_beam内的序列添加到结束中, 为1的地方需要结束了
            effective_eos_mask = next_tokens[:, :num_beams].eq(eos_token_id)  # batch_size x num_beams
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                # 是由于from_which_beam是 (batch_size, 2*num_beams)的，所以需要2*num_beams
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]  # 获取真实的从哪个beam获取的eos
            else:
                flag = False
        if flag:
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(),
                                                     eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    hypos[batch_idx].add(token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)

        # 重新组织token_ids的状态
        tokens = _next_tokens
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), tokens], dim=-1)

        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item())

        cur_len += 1

        if all(dones):
            break

    # select the best hypotheses
    tgt_len = token_ids.new(batch_size)
    best = []

    for i, hypotheses in enumerate(hypos):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = token_ids.new(batch_size, tgt_len.max().item()).fill_(0)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i] - 1] = hypo
        if eos_token_id is not None:
            decoded[i, tgt_len[i] - 1] = eos_token_id

    return decoded


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    根据top_k, top_p的值，将不满足的值置为filter_value的值

    :param torch.Tensor logits: bsz x vocab_size
    :param int top_k: 如果大于0，则只保留最top_k的词汇的概率，剩下的位置被置为filter_value
    :param int top_p: 根据(http://arxiv.org/abs/1904.09751)设置的筛选方式
    :param float filter_value:
    :param int min_tokens_to_keep: 每个sample返回的分布中有概率的词不会低于这个值
    :return:
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


if __name__ == '__main__':
    # TODO 需要检查一下greedy_generate和sample_generate是否正常工作。
    from torch import nn


    class DummyDecoder(nn.Module):
        def __init__(self, num_words):
            super().__init__()
            self.num_words = num_words

        def decode(self, tokens, past):
            batch_size = tokens.size(0)
            return torch.randn(batch_size, self.num_words), past

        def reorder_past(self, indices, past):
            return past


    num_words = 10
    batch_size = 3
    decoder = DummyDecoder(num_words)

    tokens = greedy_generate(decoder=decoder, tokens=torch.zeros(batch_size, 1).long(), past=None, max_length=20,
                             num_beams=2,
                             bos_token_id=0, eos_token_id=num_words - 1,
                             repetition_penalty=1, length_penalty=1.0)
    print(tokens)

    tokens = sample_generate(decoder, tokens=torch.zeros(batch_size, 1).long(),
                             past=None, max_length=20, num_beams=2, temperature=1.0, top_k=50,
                             top_p=1.0, bos_token_id=0, eos_token_id=num_words - 1, repetition_penalty=1.0,
                             length_penalty=1.0)
    print(tokens)

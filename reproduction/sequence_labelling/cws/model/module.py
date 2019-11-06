from torch import nn
import torch
import numpy as np

class SemiCRFShiftRelay(nn.Module):
    """
    该模块是一个decoder，但当前不支持含有tag的decode。

    """
    def __init__(self, L):
        """

        :param L: 不包含relay的长度
        """
        if L<2:
            raise RuntimeError()
        super().__init__()
        self.L = L

    def forward(self, logits, relay_logits, relay_target, relay_mask, end_seg_mask, seq_len):
        """
        relay node是接下来L个字都不是它的结束。relay的状态是往后滑动1个位置

        :param logits: batch_size x max_len x L, 当前位置往左边L个segment的分数，最后一维的0是长度为1的segment(即本身)
        :param relay_logits: batch_size x max_len, 当前位置是接下来L-1个位置都不是终点的分数
        :param relay_target: batch_size x max_len 每个位置他的segment在哪里开始的。如果超过L，则一直保持为L-1。比如长度为
            5的词，L=3, [0, 1, 2, 2, 2]
        :param relay_mask: batch_size x max_len, 在需要relay的地方为1, 长度为5的词, L=3时，为[1, 1, 1, 0, 0]
        :param end_seg_mask: batch_size x max_len, segment结束的地方为1。
        :param seq_len: batch_size, 句子的长度
        :return: loss: batch_size,
        """
        batch_size, max_len, L = logits.size()

        # 当前时刻为relay node的分数是多少
        relay_scores = logits.new_zeros(batch_size, max_len)
        # 当前时刻结束的分数是多少
        scores = logits.new_zeros(batch_size, max_len+1)
        # golden的分数
        gold_scores = relay_logits[:, 0].masked_fill(relay_mask[:, 0].eq(False), 0) + \
                                logits[:, 0, 0].masked_fill(end_seg_mask[:, 0].eq(False), 0)
        # 初始化
        scores[:, 1] = logits[:, 0, 0]
        batch_i = torch.arange(batch_size).to(logits.device).long()
        relay_scores[:, 0] = relay_logits[:, 0]
        last_relay_index = max_len - self.L
        for t in range(1, max_len):
            real_L = min(t+1, L)
            flip_logits_t = logits[:, t, :real_L].flip(dims=[1])  # flip之后低0个位置为real_L-1的segment
            # 计算relay_scores的更新
            if t<last_relay_index:
                #   (1) 从正常位置跳转
                tmp1 = relay_logits[:, t] + scores[:, t] # batch_size
                #   (2) 从relay跳转
                tmp2 = relay_logits[:, t] + relay_scores[:, t-1] # batch_size
                tmp1 = torch.stack([tmp1, tmp2], dim=0)
                relay_scores[:, t] = torch.logsumexp(tmp1, dim=0)
            # 计算scores的更新
            #  (1)从之前的位置跳转过来的
            tmp1 = scores[:, t-real_L+1:t+1] + flip_logits_t  # batch_size x L
            if t>self.L-1:
                #  (2)从relay跳转过来的
                tmp2 = relay_scores[:, t-self.L] # batch_size
                tmp2 = tmp2 + flip_logits_t[:, 0] # batch_size
                tmp1 = torch.cat([tmp1, tmp2.unsqueeze(-1)], dim=-1)
            scores[:, t+1] = torch.logsumexp(tmp1, dim=-1) # 更新当前时刻的分数

            # 计算golden
            seg_i = relay_target[:, t] # batch_size
            gold_segment_scores = logits[:, t][(batch_i, seg_i)].masked_fill(end_seg_mask[:, t].eq(False), 0) # batch_size, 后向从0到L长度的segment的分数
            relay_score = relay_logits[:, t].masked_fill(relay_mask[:, t].eq(False), 0)
            gold_scores = gold_scores + relay_score + gold_segment_scores
        all_scores = scores.gather(dim=1, index=seq_len.unsqueeze(1)).squeeze(1) # batch_size
        return all_scores - gold_scores

    def predict(self, logits, relay_logits, seq_len):
        """
        relay node是接下来L个字都不是它的结束。relay的状态是往后滑动L-1个位置

        :param logits: batch_size x max_len x L, 当前位置左边L个segment的分数，最后一维的0是长度为1的segment(即本身)
        :param relay_logits: batch_size x max_len, 当前位置是接下来L-1个位置都不是终点的分数
        :param seq_len: batch_size, 句子的长度
        :return: pred: batch_size x max_len以该点开始的segment的(长度-1); pred_mask为1的地方预测有segment开始
        """
        batch_size, max_len, L = logits.size()
        # 当前时刻为relay node的分数是多少
        max_relay_scores = logits.new_zeros(batch_size, max_len)
        relay_bt = seq_len.new_zeros(batch_size, max_len)  # 当前结果是否来自于relay的结果
        # 当前时刻结束的分数是多少
        max_scores = logits.new_zeros(batch_size, max_len+1)
        bt = seq_len.new_zeros(batch_size, max_len)
        # 初始化
        max_scores[:, 1] = logits[:, 0, 0]
        max_relay_scores[:, 0] = relay_logits[:, 0]
        last_relay_index = max_len - self.L
        for t in range(1, max_len):
            real_L = min(t+1, L)
            flip_logits_t = logits[:, t, :real_L].flip(dims=[1])  # flip之后低0个位置为real_L-1的segment
            # 计算relay_scores的更新
            if t<last_relay_index:
                #  (1) 从正常位置跳转
                tmp1 = relay_logits[:, t] + max_scores[:, t]
                #   (2) 从relay跳转
                tmp2 = relay_logits[:, t] + max_relay_scores[:, t-1] # batch_size
                # 每个sample的倒数L位不能是relay了
                tmp2 = tmp2.masked_fill(seq_len.le(t+L), float('-inf'))
                mask_i = tmp1.lt(tmp2) # 为1的位置为relay跳转
                relay_bt[:, t].masked_fill_(mask_i, 1)
                max_relay_scores[:, t] = torch.max(tmp1, tmp2)

            # 计算scores的更新
            #  (1)从之前的位置跳转过来的
            tmp1 = max_scores[:, t-real_L+1:t+1] + flip_logits_t  # batch_size x L
            tmp1 = tmp1.flip(dims=[1]) # 0的位置代表长度为1的segment
            if self.L-1<t:
                #  (2)从relay跳转过来的
                tmp2 = max_relay_scores[:, t-self.L] # batch_size
                tmp2 = tmp2 + flip_logits_t[:, 0]
                tmp1 = torch.cat([tmp1, tmp2.unsqueeze(-1)], dim=-1)
            # 看哪个更大
            max_score, pt = torch.max(tmp1, dim=1)
            max_scores[:, t+1] = max_score
            # mask_i = pt.ge(self.L)
            bt[:, t] = pt # 假设L=3, 那么对于0,1,2,3分别代表的是[t, t], [t-1, t], [t-2, t], [t-self.L(relay), t]
        # 需要把结果decode出来
        pred = np.zeros((batch_size, max_len), dtype=int)
        pred_mask = np.zeros((batch_size, max_len), dtype=int)
        seq_len = seq_len.tolist()
        bt = bt.tolist()
        relay_bt = relay_bt.tolist()
        for b in range(batch_size):
            seq_len_i = seq_len[b]
            bt_i = bt[b][:seq_len_i]
            relay_bt_i = relay_bt[b][:seq_len_i]
            j = seq_len_i - 1
            assert relay_bt_i[j]!=1
            while j>-1:
                if bt_i[j]==self.L:
                    seg_start_pos = j
                    j = j-self.L
                    while relay_bt_i[j]!=0 and j>-1:
                        j = j - 1
                    pred[b, j] = seg_start_pos - j
                    pred_mask[b, j] = 1
                else:
                    length = bt_i[j]
                    j = j - bt_i[j]
                    pred_mask[b, j] = 1
                    pred[b, j] = length
                j = j - 1

        return torch.LongTensor(pred).to(logits.device), torch.LongTensor(pred_mask).to(logits.device)



class FeatureFunMax(nn.Module):
    def __init__(self, hidden_size:int, L:int):
        """
        用于计算semi-CRF特征的函数。给定batch_size x max_len x hidden_size形状的输入，输出为batch_size x max_len x L的
        分数，以及batch_size x max_len的relay的分数。两者的区别参考论文 TODO 补充

        :param hidden_size: 输入特征的维度大小
        :param L: 不包含relay node的segment的长度大小。
        """
        super().__init__()

        self.end_fc = nn.Linear(hidden_size, 1, bias=False)
        self.whole_w = nn.Parameter(torch.randn(L, hidden_size))
        self.relay_fc = nn.Linear(hidden_size, 1)
        self.length_bias = nn.Parameter(torch.randn(L))
        self.L = L
    def forward(self, logits):
        """

        :param logits: batch_size x max_len x hidden_size
        :return: batch_size x max_len x L # 最后一维为左边segment的分数，0处为长度为1的segment
                 batch_size x max_len, # 当前位置是接下来L-1个位置都不是终点的分数

        """
        batch_size, max_len, hidden_size = logits.size()
        # start_scores = self.start_fc(logits) # batch_size x max_len x 1 # 每个位置作为start的分数
        tmp = logits.new_zeros(batch_size, max_len+self.L-1, hidden_size)
        tmp[:, -max_len:] = logits
        # batch_size x max_len x hidden_size x (self.L) -> batch_size x max_len x (self.L) x hidden_size
        start_logits = tmp.unfold(dimension=1, size=self.L, step=1).transpose(2, 3).flip(dims=[2])
        end_scores = self.end_fc(logits) # batch_size x max_len x 1
        # 计算relay的特征
        relay_tmp = logits.new_zeros(batch_size, max_len, hidden_size)
        relay_tmp[:, :-self.L] = logits[:, self.L:]
        # batch_size x max_len x hidden_size
        relay_logits_max = torch.max(relay_tmp, logits) # end - start
        logits_max = torch.max(logits.unsqueeze(2), start_logits) # batch_size x max_len x L x hidden_size
        whole_scores = (logits_max*self.whole_w).sum(dim=-1) # batch_size x max_len x self.L
        # whole_scores = self.whole_fc().squeeze(-1) # bz x max_len x self.L
        # batch_size x max_len
        relay_scores = self.relay_fc(relay_logits_max).squeeze(-1)
        return whole_scores+end_scores+self.length_bias.view(1, 1, -1), relay_scores

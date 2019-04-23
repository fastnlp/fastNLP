from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fastNLP.core.losses import LossFunc
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import seq_lens_to_masks
from fastNLP.models.base_model import BaseModel
from fastNLP.modules.dropout import TimestepDropout
from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.modules.encoder.variational_rnn import VarLSTM
from fastNLP.modules.utils import initial_parameter
from fastNLP.modules.utils import seq_mask


def _mst(scores):
    """
    with some modification to support parser output for MST decoding
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692
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
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py
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
                if not(w != v):
                    break
            _SCCs.append(SCC)

    for v in vertices:
        if v not in _indices:
            _strongconnect(v)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


class GraphParser(BaseModel):
    """
    基于图的parser base class, 支持贪婪解码和最大生成树解码
    """
    def __init__(self):
        super(GraphParser, self).__init__()

    @staticmethod
    def greedy_decoder(arc_matrix, mask=None):
        """
        贪心解码方式, 输入图, 输出贪心解码的parsing结果, 不保证合法的构成树

        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        _, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix + torch.diag(arc_matrix.new(seq_len).fill_(-np.inf))
        flip_mask = (mask == 0).byte()
        matrix.masked_fill_(flip_mask.unsqueeze(1), -np.inf)
        _, heads = torch.max(matrix, dim=2)
        if mask is not None:
            heads *= mask.long()
        return heads

    @staticmethod
    def mst_decoder(arc_matrix, mask=None):
        """
        用最大生成树算法, 计算parsing结果, 保证输出合法的树结构

        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        batch_size, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix.clone()
        ans = matrix.new_zeros(batch_size, seq_len).long()
        lens = (mask.long()).sum(1) if mask is not None else torch.zeros(batch_size) + seq_len
        for i, graph in enumerate(matrix):
            len_i = lens[i]
            ans[i, :len_i] = torch.as_tensor(_mst(graph.detach()[:len_i, :len_i].cpu().numpy()), device=ans.device)
        if mask is not None:
            ans *= mask.long()
        return ans


class ArcBiaffine(nn.Module):
    """
    Biaffine Dependency Parser 的子模块, 用于构建预测边的图

    :param hidden_size: 输入的特征维度
    :param bias: 是否使用bias. Default: ``True``
    """
    def __init__(self, hidden_size, bias=True):
        super(ArcBiaffine, self).__init__()
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.has_bias = bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        initial_parameter(self)

    def forward(self, head, dep):
        """

        :param head: arc-head tensor [batch, length, hidden]
        :param dep: arc-dependent tensor [batch, length, hidden]
        :return output: tensor [bacth, length, length]
        """
        output = dep.matmul(self.U)
        output = output.bmm(head.transpose(-1, -2))
        if self.has_bias:
            output += head.matmul(self.bias).unsqueeze(1)
        return output


class LabelBilinear(nn.Module):
    """
    Biaffine Dependency Parser 的子模块, 用于构建预测边类别的图

    :param in1_features: 输入的特征1维度
    :param in2_features: 输入的特征2维度
    :param num_label: 边类别的个数
    :param bias: 是否使用bias. Default: ``True``
    """
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False)

    def forward(self, x1, x2):
        """

        :param x1: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param x2: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(x1, x2)
        output += self.lin(torch.cat([x1, x2], dim=2))
        return output

class BiaffineParser(GraphParser):
    """Biaffine Dependency Parser 实现.
    论文参考 ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .

    :param word_vocab_size: 单词词典大小
    :param word_emb_dim: 单词词嵌入向量的维度
    :param pos_vocab_size: part-of-speech 词典大小
    :param pos_emb_dim: part-of-speech 向量维度
    :param num_label: 边的类别个数
    :param rnn_layers: rnn encoder的层数
    :param rnn_hidden_size: rnn encoder 的隐状态维度
    :param arc_mlp_size: 边预测的MLP维度
    :param label_mlp_size: 类别预测的MLP维度
    :param dropout: dropout概率.
    :param encoder: encoder类别, 可选 ('lstm', 'var-lstm', 'transformer'). Default: lstm
    :param use_greedy_infer: 是否在inference时使用贪心算法.
        若 ``False`` , 使用更加精确但相对缓慢的MST算法. Default: ``False``
    """
    def __init__(self,
                word_vocab_size,
                word_emb_dim,
                pos_vocab_size,
                pos_emb_dim,
                num_label,
                rnn_layers=1,
                rnn_hidden_size=200,
                arc_mlp_size=100,
                label_mlp_size=100,
                dropout=0.3,
                encoder='lstm',
                use_greedy_infer=False):
        super(BiaffineParser, self).__init__()
        rnn_out_size = 2 * rnn_hidden_size
        word_hid_dim = pos_hid_dim = rnn_hidden_size
        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=pos_vocab_size, embedding_dim=pos_emb_dim)
        self.word_fc = nn.Linear(word_emb_dim, word_hid_dim)
        self.pos_fc = nn.Linear(pos_emb_dim, pos_hid_dim)
        self.word_norm = nn.LayerNorm(word_hid_dim)
        self.pos_norm = nn.LayerNorm(pos_hid_dim)
        self.encoder_name = encoder
        self.max_len = 512
        if encoder == 'var-lstm':
            self.encoder = VarLSTM(input_size=word_hid_dim + pos_hid_dim,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_layers,
                                   bias=True,
                                   batch_first=True,
                                   input_dropout=dropout,
                                   hidden_dropout=dropout,
                                   bidirectional=True)
        elif encoder == 'lstm':
            self.encoder = nn.LSTM(input_size=word_hid_dim + pos_hid_dim,
                                   hidden_size=rnn_hidden_size,
                                   num_layers=rnn_layers,
                                   bias=True,
                                   batch_first=True,
                                   dropout=dropout,
                                   bidirectional=True)
        elif encoder == 'transformer':
            n_head = 16
            d_k = d_v = int(rnn_out_size / n_head)
            if (d_k * n_head) != rnn_out_size:
                raise ValueError('unsupported rnn_out_size: {} for transformer'.format(rnn_out_size))
            self.position_emb = nn.Embedding(num_embeddings=self.max_len,
                                             embedding_dim=rnn_out_size,)
            self.encoder = TransformerEncoder(num_layers=rnn_layers,
                                              model_size=rnn_out_size,
                                              inner_size=1024,
                                              key_size=d_k,
                                              value_size=d_v,
                                              num_head=n_head,
                                              dropout=dropout,)
        else:
            raise ValueError('unsupported encoder type: {}'.format(encoder))

        self.mlp = nn.Sequential(nn.Linear(rnn_out_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                          nn.ELU(),
                                          TimestepDropout(p=dropout),)
        self.arc_mlp_size = arc_mlp_size
        self.label_mlp_size = label_mlp_size
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, num_label, bias=True)
        self.use_greedy_infer = use_greedy_infer
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                continue
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 0.1)
                nn.init.constant_(m.bias, 0)
            else:
                for p in m.parameters():
                    nn.init.normal_(p, 0, 0.1)

    def forward(self, words1, words2, seq_len, gold_heads=None):
        """模型forward阶段

        :param words1: [batch_size, seq_len] 输入word序列
        :param words2: [batch_size, seq_len] 输入pos序列
        :param seq_len: [batch_size, seq_len] 输入序列长度
        :param gold_heads: [batch_size, seq_len] 输入真实标注的heads, 仅在训练阶段有效,
            用于训练label分类器. 若为 ``None`` , 使用预测的heads输入到label分类器
            Default: ``None``
        :return dict: parsing结果::

            arc_pred: [batch_size, seq_len, seq_len] 边预测logits
            label_pred: [batch_size, seq_len, num_label] label预测logits
            mask: [batch_size, seq_len] 预测结果的mask
            head_pred: [batch_size, seq_len] heads的预测结果, 在 ``gold_heads=None`` 时预测
        """
        # prepare embeddings
        batch_size, length = words1.shape
        # print('forward {} {}'.format(batch_size, seq_len))

        # get sequence mask
        mask = seq_mask(seq_len, length).long()

        word = self.word_embedding(words1) # [N,L] -> [N,L,C_0]
        pos = self.pos_embedding(words2) # [N,L] -> [N,L,C_1]

        word, pos = self.word_fc(word), self.pos_fc(pos)
        word, pos = self.word_norm(word), self.pos_norm(pos)
        x = torch.cat([word, pos], dim=2) # -> [N,L,C]

        # encoder, extract features
        if self.encoder_name.endswith('lstm'):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
            x = x[sort_idx]
            x = nn.utils.rnn.pack_padded_sequence(x, sort_lens, batch_first=True)
            feat, _ = self.encoder(x) # -> [N,L,C]
            feat, _ = nn.utils.rnn.pad_packed_sequence(feat, batch_first=True)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            feat = feat[unsort_idx]
        else:
            seq_range = torch.arange(seq_len, dtype=torch.long, device=x.device)[None,:]
            x = x + self.position_emb(seq_range)
            feat = self.encoder(x, mask.float())

        # for arc biaffine
        # mlp, reduce dim
        feat = self.mlp(feat)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feat[:,:,:arc_sz], feat[:,:,arc_sz:2*arc_sz]
        label_dep, label_head = feat[:,:,2*arc_sz:2*arc_sz+label_sz], feat[:,:,2*arc_sz+label_sz:]

        # biaffine arc classifier
        arc_pred = self.arc_predictor(arc_head, arc_dep) # [N, L, L]

        # use gold or predicted arc to predict label
        if gold_heads is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads
        else:
            assert self.training # must be training mode
            if gold_heads is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = gold_heads

        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=words1.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep) # [N, L, num_label]
        res_dict = {'arc_pred': arc_pred, 'label_pred': label_pred, 'mask': mask}
        if head_pred is not None:
            res_dict['head_pred'] = head_pred
        return res_dict

    @staticmethod
    def loss(arc_pred, label_pred, arc_true, label_true, mask):
        """
        Compute loss.

        :param arc_pred: [batch_size, seq_len, seq_len] 边预测logits
        :param label_pred: [batch_size, seq_len, num_label] label预测logits
        :param arc_true: [batch_size, seq_len] 真实边的标注
        :param label_true: [batch_size, seq_len] 真实类别的标注
        :param mask: [batch_size, seq_len] 预测结果的mask
        :return: loss value
        """

        batch_size, seq_len, _ = arc_pred.shape
        flip_mask = (mask == 0)
        _arc_pred = arc_pred.clone()
        _arc_pred.masked_fill_(flip_mask.unsqueeze(1), -float('inf'))
        arc_logits = F.log_softmax(_arc_pred, dim=2)
        label_logits = F.log_softmax(label_pred, dim=2)
        batch_index = torch.arange(batch_size, device=arc_logits.device, dtype=torch.long).unsqueeze(1)
        child_index = torch.arange(seq_len, device=arc_logits.device, dtype=torch.long).unsqueeze(0)
        arc_loss = arc_logits[batch_index, child_index, arc_true]
        label_loss = label_logits[batch_index, child_index, label_true]

        byte_mask = flip_mask.byte()
        arc_loss.masked_fill_(byte_mask, 0)
        label_loss.masked_fill_(byte_mask, 0)
        arc_nll = -arc_loss.mean()
        label_nll = -label_loss.mean()
        return arc_nll + label_nll

    def predict(self, words1, words2, seq_len):
        """模型预测API

        :param words1: [batch_size, seq_len] 输入word序列
        :param words2: [batch_size, seq_len] 输入pos序列
        :param seq_len: [batch_size, seq_len] 输入序列长度
        :param gold_heads: [batch_size, seq_len] 输入真实标注的heads, 仅在训练阶段有效,
            用于训练label分类器. 若为 ``None`` , 使用预测的heads输入到label分类器
            Default: ``None``
        :return dict: parsing结果::

            arc_pred: [batch_size, seq_len, seq_len] 边预测logits
            label_pred: [batch_size, seq_len, num_label] label预测logits
            mask: [batch_size, seq_len] 预测结果的mask
            head_pred: [batch_size, seq_len] heads的预测结果, 在 ``gold_heads=None`` 时预测
        """
        res = self(words1, words2, seq_len)
        output = {}
        output['arc_pred'] = res.pop('head_pred')
        _, label_pred = res.pop('label_pred').max(2)
        output['label_pred'] = label_pred
        return output


class ParserLoss(LossFunc):
    """
    计算parser的loss

    :param arc_pred: [batch_size, seq_len, seq_len] 边预测logits
    :param label_pred: [batch_size, seq_len, num_label] label预测logits
    :param arc_true: [batch_size, seq_len] 真实边的标注
    :param label_true: [batch_size, seq_len] 真实类别的标注
    :param mask: [batch_size, seq_len] 预测结果的mask
    :return loss: scalar
    """
    def __init__(self, arc_pred=None, label_pred=None, arc_true=None, label_true=None):
        super(ParserLoss, self).__init__(BiaffineParser.loss,
                                                 arc_pred=arc_pred,
                                                 label_pred=label_pred,
                                                 arc_true=arc_true,
                                                 label_true=label_true)


class ParserMetric(MetricBase):
    """
    评估parser的性能

    :param arc_pred: 边预测logits
    :param label_pred: label预测logits
    :param arc_true: 真实边的标注
    :param label_true: 真实类别的标注
    :param seq_len: 序列长度
    :return dict: 评估结果::

        UAS: 不带label时, 边预测的准确率
        LAS: 同时预测边和label的准确率
    """
    def __init__(self, arc_pred=None, label_pred=None,
                 arc_true=None, label_true=None, seq_len=None):

        super().__init__()
        self._init_param_map(arc_pred=arc_pred, label_pred=label_pred,
                             arc_true=arc_true, label_true=label_true,
                             seq_len=seq_len)
        self.num_arc = 0
        self.num_label = 0
        self.num_sample = 0

    def get_metric(self, reset=True):
        res = {'UAS': self.num_arc*1.0 / self.num_sample, 'LAS': self.num_label*1.0 / self.num_sample}
        if reset:
            self.num_sample = self.num_label = self.num_arc = 0
        return res

    def evaluate(self, arc_pred, label_pred, arc_true, label_true, seq_len=None):
        """Evaluate the performance of prediction.
        """
        if seq_len is None:
            seq_mask = arc_pred.new_ones(arc_pred.size(), dtype=torch.long)
        else:
            seq_mask = seq_lens_to_masks(seq_len.long(), float=False).long()
        # mask out <root> tag
        seq_mask[:,0] = 0
        head_pred_correct = (arc_pred == arc_true).long() * seq_mask
        label_pred_correct = (label_pred == label_true).long() * head_pred_correct
        self.num_arc += head_pred_correct.sum().item()
        self.num_label += label_pred_correct.sum().item()
        self.num_sample += seq_mask.sum().item()

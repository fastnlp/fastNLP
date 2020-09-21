r"""
Biaffine Dependency Parser 的 Pytorch 实现.
"""
__all__ = [
    "BiaffineParser",
    "GraphParser"
]

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from ..core.const import Const as C
from ..core.losses import LossFunc
from ..core.metrics import MetricBase
from ..core.utils import seq_len_to_mask
from ..embeddings.utils import get_embeddings
from ..modules.dropout import TimestepDropout
from ..modules.encoder.transformer import TransformerEncoder
from ..modules.encoder.variational_rnn import VarLSTM
from ..modules.utils import initial_parameter


def _mst(scores):
    r"""
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
    r"""
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
                if not (w != v):
                    break
            _SCCs.append(SCC)
    
    for v in vertices:
        if v not in _indices:
            _strongconnect(v)
    
    return [SCC for SCC in _SCCs if len(SCC) > 1]


class GraphParser(BaseModel):
    r"""
    基于图的parser base class, 支持贪婪解码和最大生成树解码
    """
    
    def __init__(self):
        super(GraphParser, self).__init__()
    
    @staticmethod
    def greedy_decoder(arc_matrix, mask=None):
        r"""
        贪心解码方式, 输入图, 输出贪心解码的parsing结果, 不保证合法的构成树

        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        _, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix + torch.diag(arc_matrix.new(seq_len).fill_(-np.inf))
        flip_mask = mask.eq(False)
        matrix.masked_fill_(flip_mask.unsqueeze(1), -np.inf)
        _, heads = torch.max(matrix, dim=2)
        if mask is not None:
            heads *= mask.long()
        return heads
    
    @staticmethod
    def mst_decoder(arc_matrix, mask=None):
        r"""
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
    r"""
    Biaffine Dependency Parser 的子模块, 用于构建预测边的图

    """
    
    def __init__(self, hidden_size, bias=True):
        r"""
        
        :param hidden_size: 输入的特征维度
        :param bias: 是否使用bias. Default: ``True``
        """
        super(ArcBiaffine, self).__init__()
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=True)
        self.has_bias = bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        initial_parameter(self)
    
    def forward(self, head, dep):
        r"""

        :param head: arc-head tensor [batch, length, hidden]
        :param dep: arc-dependent tensor [batch, length, hidden]
        :return output: tensor [bacth, length, length]
        """
        output = dep.matmul(self.U)
        output = output.bmm(head.transpose(-1, -2))
        if self.has_bias:
            output = output + head.matmul(self.bias).unsqueeze(1)
        return output


class LabelBilinear(nn.Module):
    r"""
    Biaffine Dependency Parser 的子模块, 用于构建预测边类别的图

    """
    
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        r"""
        
        :param in1_features: 输入的特征1维度
        :param in2_features: 输入的特征2维度
        :param num_label: 边类别的个数
        :param bias: 是否使用bias. Default: ``True``
        """
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False)
    
    def forward(self, x1, x2):
        r"""

        :param x1: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param x2: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(x1, x2)
        output = output + self.lin(torch.cat([x1, x2], dim=2))
        return output


class BiaffineParser(GraphParser):
    r"""
    Biaffine Dependency Parser 实现.
    论文参考 `Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016) <https://arxiv.org/abs/1611.01734>`_ .

    """
    
    def __init__(self,
                 embed,
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
        r"""
        
        :param embed: 单词词典, 可以是 tuple, 包括(num_embedings, embedding_dim), 即
            embedding的大小和每个词的维度. 也可以传入 nn.Embedding 对象,
            此时就以传入的对象作为embedding
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
        super(BiaffineParser, self).__init__()
        rnn_out_size = 2 * rnn_hidden_size
        word_hid_dim = pos_hid_dim = rnn_hidden_size
        self.word_embedding = get_embeddings(embed)
        word_emb_dim = self.word_embedding.embedding_dim
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
                                             embedding_dim=rnn_out_size, )
            self.encoder = TransformerEncoder( num_layers=rnn_layers, d_model=rnn_out_size,
                                               n_head=n_head, dim_ff=1024, dropout=dropout)
        else:
            raise ValueError('unsupported encoder type: {}'.format(encoder))
        
        self.mlp = nn.Sequential(nn.Linear(rnn_out_size, arc_mlp_size * 2 + label_mlp_size * 2),
                                 nn.ELU(),
                                 TimestepDropout(p=dropout), )
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
    
    def forward(self, words1, words2, seq_len, target1=None):
        r"""模型forward阶段

        :param words1: [batch_size, seq_len] 输入word序列
        :param words2: [batch_size, seq_len] 输入pos序列
        :param seq_len: [batch_size, seq_len] 输入序列长度
        :param target1: [batch_size, seq_len] 输入真实标注的heads, 仅在训练阶段有效,
            用于训练label分类器. 若为 ``None`` , 使用预测的heads输入到label分类器
            Default: ``None``
        :return dict: parsing
                结果::

                    pred1: [batch_size, seq_len, seq_len] 边预测logits
                    pred2: [batch_size, seq_len, num_label] label预测logits
                    pred3: [batch_size, seq_len] heads的预测结果, 在 ``target1=None`` 时预测

        """
        # prepare embeddings
        batch_size, length = words1.shape
        # print('forward {} {}'.format(batch_size, seq_len))
        
        # get sequence mask
        mask = seq_len_to_mask(seq_len, max_len=length).long()
        
        word = self.word_embedding(words1)  # [N,L] -> [N,L,C_0]
        pos = self.pos_embedding(words2)  # [N,L] -> [N,L,C_1]
        
        word, pos = self.word_fc(word), self.pos_fc(pos)
        word, pos = self.word_norm(word), self.pos_norm(pos)
        x = torch.cat([word, pos], dim=2)  # -> [N,L,C]
        
        # encoder, extract features
        if self.encoder_name.endswith('lstm'):
            sort_lens, sort_idx = torch.sort(seq_len, dim=0, descending=True)
            x = x[sort_idx]
            x = nn.utils.rnn.pack_padded_sequence(x, sort_lens, batch_first=True)
            feat, _ = self.encoder(x)  # -> [N,L,C]
            feat, _ = nn.utils.rnn.pad_packed_sequence(feat, batch_first=True)
            _, unsort_idx = torch.sort(sort_idx, dim=0, descending=False)
            feat = feat[unsort_idx]
        else:
            seq_range = torch.arange(length, dtype=torch.long, device=x.device)[None, :]
            x = x + self.position_emb(seq_range)
            feat = self.encoder(x, mask.float())
        
        # for arc biaffine
        # mlp, reduce dim
        feat = self.mlp(feat)
        arc_sz, label_sz = self.arc_mlp_size, self.label_mlp_size
        arc_dep, arc_head = feat[:, :, :arc_sz], feat[:, :, arc_sz:2 * arc_sz]
        label_dep, label_head = feat[:, :, 2 * arc_sz:2 * arc_sz + label_sz], feat[:, :, 2 * arc_sz + label_sz:]
        
        # biaffine arc classifier
        arc_pred = self.arc_predictor(arc_head, arc_dep)  # [N, L, L]
        
        # use gold or predicted arc to predict label
        if target1 is None or not self.training:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self.greedy_decoder(arc_pred, mask)
            else:
                heads = self.mst_decoder(arc_pred, mask)
            head_pred = heads
        else:
            assert self.training  # must be training mode
            if target1 is None:
                heads = self.greedy_decoder(arc_pred, mask)
                head_pred = heads
            else:
                head_pred = None
                heads = target1
        
        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=words1.device).unsqueeze(1)
        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep)  # [N, L, num_label]
        res_dict = {C.OUTPUTS(0): arc_pred, C.OUTPUTS(1): label_pred}
        if head_pred is not None:
            res_dict[C.OUTPUTS(2)] = head_pred
        return res_dict
    
    @staticmethod
    def loss(pred1, pred2, target1, target2, seq_len):
        r"""
        计算parser的loss

        :param pred1: [batch_size, seq_len, seq_len] 边预测logits
        :param pred2: [batch_size, seq_len, num_label] label预测logits
        :param target1: [batch_size, seq_len] 真实边的标注
        :param target2: [batch_size, seq_len] 真实类别的标注
        :param seq_len: [batch_size, seq_len] 真实目标的长度
        :return loss: scalar
        """
        
        batch_size, length, _ = pred1.shape
        mask = seq_len_to_mask(seq_len, max_len=length)
        flip_mask = (mask.eq(False))
        _arc_pred = pred1.clone()
        _arc_pred = _arc_pred.masked_fill(flip_mask.unsqueeze(1), -float('inf'))
        arc_logits = F.log_softmax(_arc_pred, dim=2)
        label_logits = F.log_softmax(pred2, dim=2)
        batch_index = torch.arange(batch_size, device=arc_logits.device, dtype=torch.long).unsqueeze(1)
        child_index = torch.arange(length, device=arc_logits.device, dtype=torch.long).unsqueeze(0)
        arc_loss = arc_logits[batch_index, child_index, target1]
        label_loss = label_logits[batch_index, child_index, target2]
        
        arc_loss = arc_loss.masked_fill(flip_mask, 0)
        label_loss = label_loss.masked_fill(flip_mask, 0)
        arc_nll = -arc_loss.mean()
        label_nll = -label_loss.mean()
        return arc_nll + label_nll
    
    def predict(self, words1, words2, seq_len):
        r"""模型预测API

        :param words1: [batch_size, seq_len] 输入word序列
        :param words2: [batch_size, seq_len] 输入pos序列
        :param seq_len: [batch_size, seq_len] 输入序列长度
        :return dict: parsing
                结果::

                    pred1: [batch_size, seq_len] heads的预测结果
                    pred2: [batch_size, seq_len, num_label] label预测logits

        """
        res = self(words1, words2, seq_len)
        output = {}
        output[C.OUTPUTS(0)] = res.pop(C.OUTPUTS(2))
        _, label_pred = res.pop(C.OUTPUTS(1)).max(2)
        output[C.OUTPUTS(1)] = label_pred
        return output


class ParserLoss(LossFunc):
    r"""
    计算parser的loss

    """
    
    def __init__(self, pred1=None, pred2=None,
                 target1=None, target2=None,
                 seq_len=None):
        r"""
        
        :param pred1: [batch_size, seq_len, seq_len] 边预测logits
        :param pred2: [batch_size, seq_len, num_label] label预测logits
        :param target1: [batch_size, seq_len] 真实边的标注
        :param target2: [batch_size, seq_len] 真实类别的标注
        :param seq_len: [batch_size, seq_len] 真实目标的长度
        :return loss: scalar
        """
        super(ParserLoss, self).__init__(BiaffineParser.loss,
                                         pred1=pred1,
                                         pred2=pred2,
                                         target1=target1,
                                         target2=target2,
                                         seq_len=seq_len)


class ParserMetric(MetricBase):
    r"""
    评估parser的性能

    """
    
    def __init__(self, pred1=None, pred2=None,
                 target1=None, target2=None, seq_len=None):
        r"""
        
        :param pred1: 边预测logits
        :param pred2: label预测logits
        :param target1: 真实边的标注
        :param target2: 真实类别的标注
        :param seq_len: 序列长度
        :return dict: 评估结果::
    
            UAS: 不带label时, 边预测的准确率
            LAS: 同时预测边和label的准确率
        """
        super().__init__()
        self._init_param_map(pred1=pred1, pred2=pred2,
                             target1=target1, target2=target2,
                             seq_len=seq_len)
        self.num_arc = 0
        self.num_label = 0
        self.num_sample = 0
    
    def get_metric(self, reset=True):
        res = {'UAS': self.num_arc * 1.0 / self.num_sample, 'LAS': self.num_label * 1.0 / self.num_sample}
        if reset:
            self.num_sample = self.num_label = self.num_arc = 0
        return res
    
    def evaluate(self, pred1, pred2, target1, target2, seq_len=None):
        r"""Evaluate the performance of prediction.
        """
        if seq_len is None:
            seq_mask = pred1.new_ones(pred1.size(), dtype=torch.long)
        else:
            seq_mask = seq_len_to_mask(seq_len.long()).long()
        # mask out <root> tag
        seq_mask[:, 0] = 0
        head_pred_correct = (pred1 == target1).long() * seq_mask
        label_pred_correct = (pred2 == target2).long() * head_pred_correct
        self.num_arc += head_pred_correct.sum().item()
        self.num_label += label_pred_correct.sum().item()
        self.num_sample += seq_mask.sum().item()

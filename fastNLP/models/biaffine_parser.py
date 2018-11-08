import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import copy
import numpy as np
import torch
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from fastNLP.modules.utils import initial_parameter
from fastNLP.modules.encoder.variational_rnn import VarLSTM
from fastNLP.modules.dropout import TimestepDropout

def mst(scores):
    """
    with some modification to support parser output for MST decoding
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692
    """
    length = scores.shape[0]
    min_score = -np.inf
    mask = np.zeros((length, length))
    np.fill_diagonal(mask, -np.inf)
    scores = scores + mask
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


class GraphParser(nn.Module):
    """Graph based Parser helper class, support greedy decoding and MST(Maximum Spanning Tree) decoding
    """
    def __init__(self):
        super(GraphParser, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def _greedy_decoder(self, arc_matrix, seq_mask=None):
        _, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix + torch.diag(arc_matrix.new(seq_len).fill_(-np.inf))
        _, heads = torch.max(matrix, dim=2)
        if seq_mask is not None:
            heads *= seq_mask.long()
        return heads

    def _mst_decoder(self, arc_matrix, seq_mask=None):
        batch_size, seq_len, _ = arc_matrix.shape
        matrix = torch.zeros_like(arc_matrix).copy_(arc_matrix)
        ans = matrix.new_zeros(batch_size, seq_len).long()
        for i, graph in enumerate(matrix):
            ans[i] = torch.as_tensor(mst(graph.cpu().numpy()), device=ans.device)
        if seq_mask is not None:
            ans *= seq_mask.long()
        return ans


class ArcBiaffine(nn.Module):
    """helper module for Biaffine Dependency Parser predicting arc
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
        :param head arc-head tensor = [batch, length, emb_dim]
        :param dep arc-dependent tensor = [batch, length, emb_dim]

        :return output tensor = [bacth, length, length]
        """
        output = dep.matmul(self.U)
        output = output.bmm(head.transpose(-1, -2))
        if self.has_bias:
            output += head.matmul(self.bias).unsqueeze(1)
        return output


class LabelBilinear(nn.Module):
    """helper module for Biaffine Dependency Parser predicting label
    """
    def __init__(self, in1_features, in2_features, num_label, bias=True):
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_features, in2_features, num_label, bias=bias)
        self.lin = nn.Linear(in1_features + in2_features, num_label, bias=False)

    def forward(self, x1, x2):
        output = self.bilinear(x1, x2)
        output += self.lin(torch.cat([x1, x2], dim=2))
        return output


class BiaffineParser(GraphParser):
    """Biaffine Dependency Parser implemantation.
    refer to ` Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)
    <https://arxiv.org/abs/1611.01734>`_ .
    """
    def __init__(self,
                word_vocab_size,
                word_emb_dim,
                pos_vocab_size,
                pos_emb_dim,
                rnn_layers,
                rnn_hidden_size,
                arc_mlp_size,
                label_mlp_size,
                num_label,
                dropout,
                use_var_lstm=False,
                use_greedy_infer=False):

        super(BiaffineParser, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_emb_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=pos_vocab_size, embedding_dim=pos_emb_dim)
        if use_var_lstm:
            self.lstm = VarLSTM(input_size=word_emb_dim + pos_emb_dim,
                                hidden_size=rnn_hidden_size,
                                num_layers=rnn_layers,
                                bias=True,
                                batch_first=True,
                                input_dropout=dropout,
                                hidden_dropout=dropout,
                                bidirectional=True)
        else:
            self.lstm = nn.LSTM(input_size=word_emb_dim + pos_emb_dim,
                                hidden_size=rnn_hidden_size,
                                num_layers=rnn_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=True)

        rnn_out_size = 2 * rnn_hidden_size
        self.arc_head_mlp = nn.Sequential(nn.Linear(rnn_out_size, arc_mlp_size),
                                          nn.ELU(),
                                          TimestepDropout(p=dropout),)
        self.arc_dep_mlp = copy.deepcopy(self.arc_head_mlp)
        self.label_head_mlp = nn.Sequential(nn.Linear(rnn_out_size, label_mlp_size),
                                            nn.ELU(),
                                            TimestepDropout(p=dropout),)
        self.label_dep_mlp = copy.deepcopy(self.label_head_mlp)
        self.arc_predictor = ArcBiaffine(arc_mlp_size, bias=True)
        self.label_predictor = LabelBilinear(label_mlp_size, label_mlp_size, num_label, bias=True)
        self.normal_dropout = nn.Dropout(p=dropout)
        self.use_greedy_infer = use_greedy_infer
        initial_parameter(self)

    def forward(self, word_seq, pos_seq, seq_mask, gold_heads=None, **_):
        """
        :param word_seq: [batch_size, seq_len] sequence of word's indices
        :param pos_seq: [batch_size, seq_len] sequence of word's indices
        :param seq_mask: [batch_size, seq_len] sequence of length masks
        :param gold_heads: [batch_size, seq_len] sequence of golden heads
        :return dict: parsing results
            arc_pred: [batch_size, seq_len, seq_len]
            label_pred: [batch_size, seq_len, seq_len]
            seq_mask: [batch_size, seq_len]
            head_pred: [batch_size, seq_len] if gold_heads is not provided, predicting the heads
        """
        # prepare embeddings
        batch_size, seq_len = word_seq.shape
        # print('forward {} {}'.format(batch_size, seq_len))
        batch_range = torch.arange(start=0, end=batch_size, dtype=torch.long, device=word_seq.device).unsqueeze(1)

        # get sequence mask
        seq_mask = seq_mask.long()

        word = self.normal_dropout(self.word_embedding(word_seq)) # [N,L] -> [N,L,C_0]
        pos = self.normal_dropout(self.pos_embedding(pos_seq)) # [N,L] -> [N,L,C_1]
        x = torch.cat([word, pos], dim=2) # -> [N,L,C]

        # lstm, extract features
        feat, _ = self.lstm(x) # -> [N,L,C]

        # for arc biaffine
        # mlp, reduce dim
        arc_dep = self.arc_dep_mlp(feat)
        arc_head = self.arc_head_mlp(feat)
        label_dep = self.label_dep_mlp(feat)
        label_head = self.label_head_mlp(feat)

        # biaffine arc classifier
        arc_pred = self.arc_predictor(arc_head, arc_dep) # [N, L, L]
        flip_mask = (seq_mask == 0)
        arc_pred.masked_fill_(flip_mask.unsqueeze(1), -np.inf)

        # use gold or predicted arc to predict label
        if gold_heads is None:
            # use greedy decoding in training
            if self.training or self.use_greedy_infer:
                heads = self._greedy_decoder(arc_pred, seq_mask)
            else:
                heads = self._mst_decoder(arc_pred, seq_mask)
            head_pred = heads
        else:
            head_pred = None
            heads = gold_heads

        label_head = label_head[batch_range, heads].contiguous()
        label_pred = self.label_predictor(label_head, label_dep) # [N, L, num_label]
        res_dict = {'arc_pred': arc_pred, 'label_pred': label_pred, 'seq_mask': seq_mask}
        if head_pred is not None:
            res_dict['head_pred'] = head_pred
        return res_dict

    def loss(self, arc_pred, label_pred, head_indices, head_labels, seq_mask, **_):
        """
        Compute loss.

        :param arc_pred: [batch_size, seq_len, seq_len]
        :param label_pred: [batch_size, seq_len, seq_len]
        :param head_indices: [batch_size, seq_len]
        :param head_labels: [batch_size, seq_len]
        :param seq_mask: [batch_size, seq_len]
        :return: loss value
        """

        batch_size, seq_len, _ = arc_pred.shape
        arc_logits = F.log_softmax(arc_pred, dim=2)
        label_logits = F.log_softmax(label_pred, dim=2)
        batch_index = torch.arange(start=0, end=batch_size, device=arc_logits.device).long().unsqueeze(1)
        child_index = torch.arange(start=0, end=seq_len, device=arc_logits.device).long().unsqueeze(0)
        arc_loss = arc_logits[batch_index, child_index, head_indices]
        label_loss = label_logits[batch_index, child_index, head_labels]

        arc_loss = arc_loss[:, 1:]
        label_loss = label_loss[:, 1:]

        float_mask = seq_mask[:, 1:].float()
        length = (seq_mask.sum() - batch_size).float()
        arc_nll = -(arc_loss*float_mask).sum() / length
        label_nll = -(label_loss*float_mask).sum() / length
        return arc_nll + label_nll

    def evaluate(self, arc_pred, label_pred, head_indices, head_labels, seq_mask, **kwargs):
        """
        Evaluate the performance of prediction.

        :return dict: performance results.
            head_pred_corrct: number of correct predicted heads.
            label_pred_correct: number of correct predicted labels.
            total_tokens: number of predicted tokens
        """
        if 'head_pred' in kwargs:
            head_pred = kwargs['head_pred']
        elif self.use_greedy_infer:
            head_pred = self._greedy_decoder(arc_pred, seq_mask)
        else:
            head_pred = self._mst_decoder(arc_pred, seq_mask)

        head_pred_correct = (head_pred == head_indices).long() * seq_mask
        _, label_preds = torch.max(label_pred, dim=2)
        label_pred_correct = (label_preds == head_labels).long() * head_pred_correct
        return {"head_pred_correct": head_pred_correct.sum(dim=1),
                "label_pred_correct": label_pred_correct.sum(dim=1),
                "total_tokens": seq_mask.sum(dim=1)}

    def metrics(self, head_pred_correct, label_pred_correct, total_tokens, **_):
        """
        Compute the metrics of model

        :param head_pred_corrct: number of correct predicted heads.
        :param label_pred_correct: number of correct predicted labels.
        :param total_tokens: number of predicted tokens
        :return dict: the metrics results
            UAS: the head predicted accuracy
            LAS: the label predicted accuracy
        """
        return {"UAS": head_pred_correct.sum().float() / total_tokens.sum().float() * 100,
                "LAS": label_pred_correct.sum().float() / total_tokens.sum().float() * 100}


from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import seq_len_to_mask
import torch


class SegAppCharParseF1Metric(MetricBase):
    #
    def __init__(self, app_index):
        super().__init__()
        self.app_index = app_index

        self.parse_head_tp = 0
        self.parse_label_tp = 0
        self.rec_tol = 0
        self.pre_tol = 0

    def evaluate(self, gold_word_pairs, gold_label_word_pairs, head_preds, label_preds, seq_lens,
                 pun_masks):
        """

        max_len是不包含root的character的长度
        :param gold_word_pairs: List[List[((head_start, head_end), (dep_start, dep_end)), ...]], batch_size
        :param gold_label_word_pairs: List[List[((head_start, head_end), label, (dep_start, dep_end)), ...]], batch_size
        :param head_preds: batch_size x max_len
        :param label_preds: batch_size x max_len
        :param seq_lens:
        :param pun_masks: batch_size x
        :return:
        """
        # 去掉root
        head_preds = head_preds[:, 1:].tolist()
        label_preds = label_preds[:, 1:].tolist()
        seq_lens = (seq_lens - 1).tolist()

        # 先解码出words，POS，heads, labels, 对应的character范围
        for b in range(len(head_preds)):
            seq_len = seq_lens[b]
            head_pred = head_preds[b][:seq_len]
            label_pred = label_preds[b][:seq_len]

            words = [] # 存放[word_start, word_end)，相对起始位置，不考虑root
            heads = []
            labels = []
            ranges = []  # 对应该char是第几个word，长度是seq_len+1
            word_idx = 0
            word_start_idx = 0
            for idx, (label, head) in enumerate(zip(label_pred, head_pred)):
                ranges.append(word_idx)
                if label == self.app_index:
                    pass
                else:
                    labels.append(label)
                    heads.append(head)
                    words.append((word_start_idx, idx+1))
                    word_start_idx = idx+1
                    word_idx += 1

            head_dep_tuple = [] # head在前面
            head_label_dep_tuple = []
            for idx, head in enumerate(heads):
                span = words[idx]
                if span[0]==span[1]-1 and pun_masks[b, span[0]]:
                    continue  # exclude punctuations
                if head == 0:
                    head_dep_tuple.append((('root', words[idx])))
                    head_label_dep_tuple.append(('root', labels[idx], words[idx]))
                else:
                    head_word_idx = ranges[head-1]
                    head_word_span = words[head_word_idx]
                    head_dep_tuple.append(((head_word_span, words[idx])))
                    head_label_dep_tuple.append((head_word_span, labels[idx], words[idx]))

            gold_head_dep_tuple = set(gold_word_pairs[b])
            gold_head_label_dep_tuple = set(gold_label_word_pairs[b])

            for head_dep, head_label_dep in zip(head_dep_tuple, head_label_dep_tuple):
                if head_dep in gold_head_dep_tuple:
                    self.parse_head_tp += 1
                if head_label_dep in gold_head_label_dep_tuple:
                    self.parse_label_tp += 1
            self.pre_tol += len(head_dep_tuple)
            self.rec_tol += len(gold_head_dep_tuple)

    def get_metric(self, reset=True):
        u_p = self.parse_head_tp / self.pre_tol
        u_r = self.parse_head_tp / self.rec_tol
        u_f = 2*u_p*u_r/(1e-6 + u_p + u_r)
        l_p = self.parse_label_tp / self.pre_tol
        l_r = self.parse_label_tp / self.rec_tol
        l_f = 2*l_p*l_r/(1e-6 + l_p + l_r)

        if reset:
            self.parse_head_tp = 0
            self.parse_label_tp = 0
            self.rec_tol = 0
            self.pre_tol = 0

        return {'u_f1': round(u_f, 4), 'u_p': round(u_p, 4), 'u_r/uas':round(u_r, 4),
                'l_f1': round(l_f, 4), 'l_p': round(l_p, 4), 'l_r/las': round(l_r, 4)}


class CWSMetric(MetricBase):
    def __init__(self, app_index):
        super().__init__()
        self.app_index = app_index
        self.pre = 0
        self.rec = 0
        self.tp = 0

    def evaluate(self, seg_targets, seg_masks, label_preds, seq_lens):
        """

        :param seg_targets: batch_size x max_len, 每个位置预测的是该word的长度-1，在word结束的地方。
        :param seg_masks: batch_size x max_len，只有在word结束的地方为1
        :param label_preds: batch_size x max_len
        :param seq_lens: batch_size
        :return:
        """

        pred_masks = torch.zeros_like(seg_masks)
        pred_segs = torch.zeros_like(seg_targets)

        seq_lens = (seq_lens - 1).tolist()
        for idx, label_pred in enumerate(label_preds[:, 1:].tolist()):
            seq_len = seq_lens[idx]
            label_pred = label_pred[:seq_len]
            word_len = 0
            for l_i, label in enumerate(label_pred):
                if label==self.app_index and l_i!=len(label_pred)-1:
                    word_len += 1
                else:
                    pred_segs[idx, l_i] = word_len # 这个词的长度为word_len
                    pred_masks[idx, l_i] = 1
                    word_len = 0

        right_mask = seg_targets.eq(pred_segs) # 对长度的预测一致
        self.rec += seg_masks.sum().item()
        self.pre += pred_masks.sum().item()
        # 且pred和target在同一个地方有值
        self.tp += (right_mask.__and__(pred_masks.byte().__and__(seg_masks.byte()))).sum().item()

    def get_metric(self, reset=True):
        res = {}
        res['rec'] = round(self.tp/(self.rec+1e-6), 4)
        res['pre'] = round(self.tp/(self.pre+1e-6), 4)
        res['f1'] = round(2*res['rec']*res['pre']/(res['pre'] + res['rec'] + 1e-6), 4)

        if reset:
            self.pre = 0
            self.rec = 0
            self.tp = 0

        return res


class ParserMetric(MetricBase):
    def __init__(self, ):
        super().__init__()
        self.num_arc = 0
        self.num_label = 0
        self.num_sample = 0

    def get_metric(self, reset=True):
        res = {'UAS': round(self.num_arc*1.0 / self.num_sample, 4),
               'LAS': round(self.num_label*1.0 / self.num_sample, 4)}
        if reset:
            self.num_sample = self.num_label = self.num_arc = 0
        return res

    def evaluate(self, head_preds, label_preds, heads, labels, seq_lens=None):
        """Evaluate the performance of prediction.
        """
        if seq_lens is None:
            seq_mask = head_preds.new_ones(head_preds.size(), dtype=torch.byte)
        else:
            seq_mask = seq_len_to_mask(seq_lens.long(), float=False)
        # mask out <root> tag
        seq_mask[:, 0] = 0
        head_pred_correct = (head_preds == heads).__and__(seq_mask)
        label_pred_correct = (label_preds == labels).__and__(head_pred_correct)
        self.num_arc += head_pred_correct.float().sum().item()
        self.num_label += label_pred_correct.float().sum().item()
        self.num_sample += seq_mask.sum().item()


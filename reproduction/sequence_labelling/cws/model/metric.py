
from fastNLP.core.metrics import MetricBase


class RelayMetric(MetricBase):
    def __init__(self, pred=None, pred_mask=None, target=None, start_seg_mask=None):
        super().__init__()
        self._init_param_map(pred=pred, pred_mask=pred_mask, target=target, start_seg_mask=start_seg_mask)
        self.tp = 0
        self.rec = 0
        self.pre = 0

    def evaluate(self, pred, pred_mask, target, start_seg_mask):
        """
        给定每个batch，累计一下结果。

        :param pred: 预测的结果，为当前位置的开始的segment的（长度-1）
        :param pred_mask: 当前位置预测有segment开始
        :param target: 当前位置开始的segment的(长度-1)
        :param start_seg_mask: 当前有segment结束
        :return:
        """
        self.tp += ((pred.long().eq(target.long())).__and__(pred_mask.byte().__and__(start_seg_mask.byte()))).sum().item()
        self.rec += start_seg_mask.sum().item()
        self.pre += pred_mask.sum().item()

    def get_metric(self, reset=True):
        """
        在所有数据都计算结束之后，得到performance
        
        :param reset:
        :return:
        """
        pre = self.tp/(self.pre + 1e-12)
        rec = self.tp/(self.rec + 1e-12)
        f = 2*pre*rec/(1e-12 + pre + rec)

        if reset:
            self.tp = 0
            self.rec = 0
            self.pre = 0
            self.bigger_than_L = 0

        return {'f': round(f, 6), 'pre': round(pre, 6), 'rec': round(rec, 6)}

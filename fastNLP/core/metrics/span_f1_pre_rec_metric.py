__all__ = [
    'SpanFPreRecMetric'
]

from typing import Union, List, Optional
import warnings
from collections import defaultdict
from functools import partial

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.metric import Metric
from fastNLP.core.vocabulary import Vocabulary


def _check_tag_vocab_and_encoding_type(tag_vocab: Union[Vocabulary, dict], encoding_type: str):
    r"""
    检查vocab中的tag是否与encoding_type是匹配的

    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :param encoding_type: bio, bmes, bioes, bmeso
    :return:
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    tags = encoding_type
    for tag in tag_set:
        assert tag in tags, f"{tag} is not a valid tag in encoding type:{encoding_type}. Please check your " \
                            f"encoding_type."
        tags = tags.replace(tag, '')  # 删除该值
    if tags:  # 如果不为空，说明出现了未使用的tag
        warnings.warn(f"Tag:{tags} in encoding type:{encoding_type} is not presented in your Vocabulary. Check your "
                      "encoding_type.")


def _get_encoding_type_from_tag_vocab(tag_vocab: Union[Vocabulary, dict]) -> str:
    r"""
    给定Vocabulary自动判断是哪种类型的encoding, 支持判断bmes, bioes, bmeso, bio

    :param tag_vocab: 支持传入tag Vocabulary; 或者传入形如{0:"O", 1:"B-tag1"}，即index在前，tag在后的dict。
    :return:
    """
    tag_set = set()
    unk_token = '<unk>'
    pad_token = '<pad>'
    if isinstance(tag_vocab, Vocabulary):
        unk_token = tag_vocab.unknown
        pad_token = tag_vocab.padding
        tag_vocab = tag_vocab.idx2word
    for idx, tag in tag_vocab.items():
        if tag in (unk_token, pad_token):
            continue
        tag = tag[:1].lower()
        tag_set.add(tag)

    bmes_tag_set = set('bmes')
    if tag_set == bmes_tag_set:
        return 'bmes'
    bio_tag_set = set('bio')
    if tag_set == bio_tag_set:
        return 'bio'
    bmeso_tag_set = set('bmeso')
    if tag_set == bmeso_tag_set:
        return 'bmeso'
    bioes_tag_set = set('bioes')
    if tag_set == bioes_tag_set:
        return 'bioes'
    raise RuntimeError("encoding_type cannot be inferred automatically. Only support "
                       "'bio', 'bmes', 'bmeso', 'bioes' type.")


def _bmes_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['S-song', 'B-singer', 'M-singer', 'E-singer', 'S-moive', 'S-actor']。
    返回[('song', (0, 1)), ('singer', (1, 4)), ('moive', (4, 5)), ('actor', (5, 6))] (左闭右开区间)
    也可以是单纯的['S', 'B', 'M', 'E', 'B', 'M', 'M',...]序列

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bmeso_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bioes_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bioes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bioes_tag, label = tag[:1], tag[2:]
        if bioes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bioes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bioes_tag = bioes_tag
    return [(span[0], (span[1][0], span[1][1] + 1))
            for span in spans
            if span[0] not in ignore_labels
            ]


def _bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == 'b':
            spans.append((label, [idx, idx]))
        elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == 'o':  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]


def _compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * pre * rec / (beta_square * pre + rec + 1e-13)

    return f, pre, rec


class SpanFPreRecMetric(Metric):

    def __init__(self, tag_vocab: Vocabulary, encoding_type: str = None, ignore_labels: List[str] = None,
                 only_gross: bool = True, f_type='micro',
                 beta=1, backend: Union[str, Backend, None] = 'auto', aggregate_when_get_metric: bool = None) -> None:
        r"""

        :param tag_vocab: 标签的 :class:`~fastNLP.Vocabulary` 。支持的标签为"B"(没有label)；或"B-xxx"(xxx为某种label，比如POS中的NN)，
            在解码时，会将相同xxx的认为是同一个label，比如['B-NN', 'E-NN']会被合并为一个'NN'.
        :param str pred: 用该key在evaluate()时从传入dict中取出prediction数据。 为None，则使用 `pred` 取数据
        :param str target: 用该key在evaluate()时从传入dict中取出target数据。 为None，则使用 `target` 取数据
        :param str seq_len: 用该key在evaluate()时从传入dict中取出sequence length数据。为None，则使用 `seq_len` 取数据。
        :param str encoding_type: 目前支持bio, bmes, bmeso, bioes。默认为None，通过tag_vocab自动判断.
        :param list ignore_labels: str 组成的list. 这个list中的class不会被用于计算。例如在POS tagging时传入['NN']，则不会计算'NN'个label
        :param bool only_gross: 是否只计算总的f1, precision, recall的值；如果为False，不仅返回总的f1, pre, rec, 还会返回每个label的f1, pre, rec
        :param str f_type: `micro` 或 `macro` . `micro` :通过先计算总体的TP，FN和FP的数量，再计算f, precision, recall; `macro` : 分布计算每个类别的f, precision, recall，然后做平均（各类别f的权重相同）
        :param float beta: f_beta分数， :math:`f_{beta} = \frac{(1 + {beta}^{2})*(pre*rec)}{({beta}^{2}*pre + rec)}` . 常用为 `beta=0.5, 1, 2` 若为0.5则精确率的权重高于召回率；若为1，则两者平等；若为2，则召回率权重高于精确率。
        :param str backend: 目前支持四种类型的backend, ['auto', 'torch', 'paddle', 'jittor']。其中 auto 表示根据实际调用 Metric.update()
            函数时传入的参数决定具体的 backend ，一般情况下直接使用 'auto' 即可。
        :param bool aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字聚合后再得到metric，
            当 backend 不支持分布式时，该参数无意义。如果为 None ，将在 Evaluator 中根据 sampler 是否使用分布式进行自动设置。
        """
        super(SpanFPreRecMetric, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))
        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if encoding_type:
            encoding_type = encoding_type.lower()
            _check_tag_vocab_and_encoding_type(tag_vocab, encoding_type)
            self.encoding_type = encoding_type
        else:
            self.encoding_type = _get_encoding_type_from_tag_vocab(tag_vocab)

        if self.encoding_type == 'bmes':
            self.tag_to_span_func = _bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = _bio_tag_to_spans
        elif self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = _bioes_tag_to_spans
        else:
            raise ValueError("Only support 'bio', 'bmes', 'bmeso', 'bioes' type.")

        self.ignore_labels = ignore_labels
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.only_gross = only_gross
        self.tag_vocab = tag_vocab

        self._true_positives = {}
        self._false_positives = {}
        self._false_negatives = {}
        for word, _ in tag_vocab:
            word = word.lower()
            if word != 'o':
                word = word[2:]
            if word in self._true_positives:
                continue
            self._true_positives[word] = self.register_element(name=f'tp_{word}', aggregate_method='sum', backend=backend)
            self._false_negatives[word] = self.register_element(name=f'fn_{word}', aggregate_method='sum', backend=backend)
            self._false_positives[word] = self.register_element(name=f'fp_{word}', aggregate_method='sum', backend=backend)

    def get_metric(self) -> dict:
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(self._false_positives.keys())
            tags.update(self._true_positives.keys())
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag].get_scalar()
                fn = self._false_negatives[tag].get_scalar()
                fp = self._false_positives[tag].get_scalar()
                if tp == fn == fp == 0:
                    continue

                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            tp, fn, fp = [], [], []
            for val in self._true_positives.values():
                tp.append(val.get_scalar())
            for val in self._false_negatives.values():
                fn.append(val.get_scalar())
            for val in self._false_positives.values():
                fp.append(val.get_scalar())
            f, pre, rec = _compute_f_pre_rec(self.beta_square,
                                             sum(tp),
                                             sum(fn),
                                             sum(fp))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def update(self, pred, target, seq_len: Optional[List] = None) -> None:
        r"""update函数将针对一个批次的预测结果做评价指标的累计

        :param pred: [batch, seq_len] 或者 [batch, seq_len, len(tag_vocab)], 预测的结果
        :param target: [batch, seq_len], 真实值
        :param seq_len: [batch] 文本长度标记
        :return:
        """
        pred = self.tensor2numpy(pred)
        target = self.tensor2numpy(target)

        if pred.ndim == target.ndim and target.ndim == 2:
            pass

        elif pred.ndim == target.ndim + 1 and target.ndim == 2:
            num_classes = pred.shape[-1]
            pred = pred.argmax(axis=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"when pred have size:{pred.ndim}, target should have size: {pred.ndim} or "
                               f"{pred.shape[:-1]}, got {target.ndim}.")

        batch_size = pred.shape[0]
        pred = pred.tolist()
        target = target.tolist()
        for i in range(batch_size):
            pred_tags = pred[i][:int(seq_len[i])]
            gold_tags = target[i][:int(seq_len[i])]

            pred_str_tags = [self.tag_vocab.to_word(tag) for tag in pred_tags]
            gold_str_tags = [self.tag_vocab.to_word(tag) for tag in gold_tags]

            pred_spans = self.tag_to_span_func(pred_str_tags, ignore_labels=self.ignore_labels)
            gold_spans = self.tag_to_span_func(gold_str_tags, ignore_labels=self.ignore_labels)

            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1

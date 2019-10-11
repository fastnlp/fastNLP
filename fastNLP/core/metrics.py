"""
metrics 模块实现了 fastNLP 所需的各种常用衡量指标，一般做为 :class:`~fastNLP.Trainer` 的参数使用。

"""
__all__ = [
    "MetricBase",
    "AccuracyMetric",
    "SpanFPreRecMetric",
    "CMRC2018Metric"
]

import inspect
import warnings
from abc import abstractmethod
from collections import defaultdict
from typing import Union
import re

import numpy as np
import torch

from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _get_func_signature
from .utils import seq_len_to_mask
from .vocabulary import Vocabulary


class MetricBase(object):
    """
    所有metrics的基类,所有的传入到Trainer, Tester的Metric需要继承自该对象，需要覆盖写入evaluate(), get_metric()方法。
    
        evaluate(xxx)中传入的是一个batch的数据。
        
        get_metric(xxx)当所有数据处理完毕，调用该方法得到最终的metric值
        
    以分类问题中，Accuracy计算为例
    假设model的forward返回dict中包含 `pred` 这个key, 并且该key需要用于Accuracy::
    
        class Model(nn.Module):
            def __init__(xxx):
                # do something
            def forward(self, xxx):
                # do something
                return {'pred': pred, 'other_keys':xxx} # pred's shape: batch_size x num_classes
                
    假设dataset中 `label` 这个field是需要预测的值，并且该field被设置为了target
    对应的AccMetric可以按如下的定义, version1, 只使用这一次::
    
        class AccMetric(MetricBase):
            def __init__(self):
                super().__init__()
    
                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0
    
            def evaluate(self, label, pred): # 这里的名称需要和dataset中target field与model返回的key是一样的，不然找不到对应的value
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()
    
            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    version2，如果需要复用Metric，比如下一次使用AccMetric时，dataset中目标field不叫label而叫y，或者model的输出不是pred::
    
        class AccMetric(MetricBase):
            def __init__(self, label=None, pred=None):
                # 假设在另一场景使用时，目标field叫y，model给出的key为pred_y。则只需要在初始化AccMetric时，
                #   acc_metric = AccMetric(label='y', pred='pred_y')即可。
                # 当初始化为acc_metric = AccMetric()，即label=None, pred=None, fastNLP会直接使用'label', 'pred'作为key去索取对
                #   应的的值
                super().__init__()
                self._init_param_map(label=label, pred=pred) # 该方法会注册label和pred. 仅需要注册evaluate()方法会用到的参数名即可
                # 如果没有注册该则效果与version1就是一样的
    
                # 根据你的情况自定义指标
                self.corr_num = 0
                self.total = 0
    
            def evaluate(self, label, pred): # 这里的参数名称需要和self._init_param_map()注册时一致。
                # dev或test时，每个batch结束会调用一次该方法，需要实现如何根据每个batch累加metric
                self.total += label.size(0)
                self.corr_num += label.eq(pred).sum().item()
    
            def get_metric(self, reset=True): # 在这里定义如何计算metric
                acc = self.corr_num/self.total
                if reset: # 是否清零以便重新计算
                    self.corr_num = 0
                    self.total = 0
                return {'acc': acc} # 需要返回一个dict，key为该metric的名称，该名称会显示到Trainer的progress bar中


    ``MetricBase`` 将会在输入的字典 ``pred_dict`` 和 ``target_dict`` 中进行检查.
    ``pred_dict`` 是模型当中 ``forward()`` 函数或者 ``predict()`` 函数的返回值.
    ``target_dict`` 是DataSet当中的ground truth, 判定ground truth的条件是field的 ``is_target`` 被设置为True.

    ``MetricBase`` 会进行以下的类型检测:

    1. self.evaluate当中是否有varargs, 这是不支持的.
    2. self.evaluate当中所需要的参数是否既不在 ``pred_dict`` 也不在 ``target_dict`` .
    3. self.evaluate当中所需要的参数是否既在 ``pred_dict`` 也在 ``target_dict`` .

    除此以外，在参数被传入self.evaluate以前，这个函数会检测 ``pred_dict`` 和 ``target_dict`` 当中没有被用到的参数
    如果kwargs是self.evaluate的参数，则不会检测


    self.evaluate将计算一个批次(batch)的评价指标，并累计。 没有返回值
    self.get_metric将统计当前的评价指标并返回评价结果, 返回值需要是一个dict, key是指标名称，value是指标的值

    """

    def __init__(self):
        self._param_map = {}  # key is param in function, value is input param.
        self._checked = False
        self._metric_name = self.__class__.__name__

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def set_metric_name(self, name: str):
        """
        设置metric的名称，默认是Metric的class name.

        :param str name:
        :return: self
        """
        self._metric_name = name
        return self

    def get_metric_name(self):
        """
        返回metric的名称
        
        :return:
        """
        return self._metric_name

    def _init_param_map(self, key_map=None, **kwargs):
        """检查key_map和其他参数map，并将这些映射关系添加到self._param_map

        :param dict key_map: 表示key的映射关系
        :param kwargs: key word args里面的每一个的键-值对都会被构造成映射关系
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self._param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self._param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self._param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self._param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and _param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

    def _fast_param_map(self, pred_dict, target_dict):
        """Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element

        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not {}, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        if len(self._param_map) == 2 and len(pred_dict) == 1 and len(target_dict) == 1:
            fast_param['pred'] = list(pred_dict.values())[0]
            fast_param['target'] = list(target_dict.values())[0]
            return fast_param
        return fast_param

    def __call__(self, pred_dict, target_dict):
        """
        这个方法会调用self.evaluate 方法.
        在调用之前，会进行以下检测:
            1. self.evaluate当中是否有varargs, 这是不支持的.
            2. self.evaluate当中所需要的参数是否既不在``pred_dict``也不在``target_dict``.
            3. self.evaluate当中所需要的参数是否既在``pred_dict``也在``target_dict``.

            除此以外，在参数被传入self.evaluate以前，这个函数会检测``pred_dict``和``target_dict``当中没有被用到的参数
            如果kwargs是self.evaluate的参数，则不会检测
        :param pred_dict: 模型的forward函数或者predict函数返回的dict
        :param target_dict: DataSet.batch_y里的键-值对所组成的dict(即is_target=True的fields的内容)
        :return:
        """

        fast_param = self._fast_param_map(pred_dict, target_dict)
        if fast_param:
            self.evaluate(**fast_param)
            return

        if not self._checked:
            if not callable(self.evaluate):
                raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.evaluate)}.")

            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        for input_arg, mapped_arg in self._reverse_param_map.items():
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]

        # missing
        if not self._checked:
            duplicated = []
            for input_arg, mapped_arg in self._reverse_param_map.items():
                if input_arg in pred_dict and input_arg in target_dict:
                    duplicated.append(input_arg)
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self._param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                         f"in `{self.__class__.__name__}`)"

            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.evaluate))
            self._checked = True
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)

        return


class AccuracyMetric(MetricBase):
    """
    准确率Metric（其它的Metric参见 :mod:`fastNLP.core.metrics` ）
    """

    def __init__(self, pred=None, target=None, seq_len=None):
        """
        
        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param seq_len: 参数映射表中 `seq_len` 的映射关系，None表示映射关系为 `seq_len` -> `seq_len`
        """

        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.total = 0
        self.acc_count = 0

    def evaluate(self, pred, target, seq_len=None):
        """
        evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param torch.Tensor pred: 预测的tensor, tensor的形状可以是torch.Size([B,]), torch.Size([B, n_classes]),
                torch.Size([B, max_len]), 或者torch.Size([B, max_len, n_classes])
        :param torch.Tensor target: 真实值的tensor, tensor的形状可以是Element's can be: torch.Size([B,]),
                torch.Size([B,]), torch.Size([B, max_len]), 或者torch.Size([B, max_len])
        :param torch.Tensor seq_len: 序列长度标记, 标记的形状可以是None, None, torch.Size([B]), 或者torch.Size([B]).
                如果mask也被传进来的话seq_len会被忽略.

        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        if pred.dim() == target.dim():
            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad when calculate accuracy.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(0), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        """
        get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果.

        :param bool reset: 在调用完get_metric后是否清空评价指标统计量.
        :return dict evaluate_result: {"acc": float}
        """
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


def _bmes_tag_to_spans(tags, ignore_labels=None):
    """
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
    """
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
    """
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
    """
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


def _get_encoding_type_from_tag_vocab(tag_vocab: Union[Vocabulary, dict]) -> str:
    """
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


def _check_tag_vocab_and_encoding_type(tag_vocab: Union[Vocabulary, dict], encoding_type: str):
    """
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


class SpanFPreRecMetric(MetricBase):
    r"""
    在序列标注问题中，以span的方式计算F, pre, rec.
    比如中文Part of speech中，会以character的方式进行标注，句子 `中国在亚洲` 对应的POS可能为(以BMES为例)
    ['B-NN', 'E-NN', 'S-DET', 'B-NN', 'E-NN']。该metric就是为类似情况下的F1计算。
    最后得到的metric结果为::
    
        {
            'f': xxx, # 这里使用f考虑以后可以计算f_beta值
            'pre': xxx,
            'rec':xxx
        }
    
    若only_gross=False, 即还会返回各个label的metric统计值::
    
        {
            'f': xxx,
            'pre': xxx,
            'rec':xxx,
            'f-label': xxx,
            'pre-label': xxx,
            'rec-label':xxx,
            ...
        }
    """

    def __init__(self, tag_vocab, pred=None, target=None, seq_len=None, encoding_type=None, ignore_labels=None,
                 only_gross=True, f_type='micro', beta=1):
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
        """

        if not isinstance(tag_vocab, Vocabulary):
            raise TypeError("tag_vocab can only be fastNLP.Vocabulary, not {}.".format(type(tag_vocab)))
        if f_type not in ('micro', 'macro'):
            raise ValueError("f_type only supports `micro` or `macro`', got {}.".format(f_type))

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

        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)

        self.tag_vocab = tag_vocab

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, pred, target, seq_len):
        """evaluate函数将针对一个批次的预测结果做评价指标的累计

        :param pred: [batch, seq_len] 或者 [batch, seq_len, len(tag_vocab)], 预测的结果
        :param target: [batch, seq_len], 真实值
        :param seq_len: [batch] 文本长度标记
        :return:
        """
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.size() == target.size() and len(target.size()) == 2:
            pass
        elif len(pred.size()) == len(target.size()) + 1 and len(target.size()) == 2:
            num_classes = pred.size(-1)
            pred = pred.argmax(dim=-1)
            if (target >= num_classes).any():
                raise ValueError("A gold label passed to SpanBasedF1Metric contains an "
                                 "id >= {}, the number of classes.".format(num_classes))
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        batch_size = pred.size(0)
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

    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
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
            f, pre, rec = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """

        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, pre, rec)
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)

        return f, pre, rec


def _prepare_metrics(metrics):
    """

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


def _accuracy_topk(y_true, y_prob, k=1):
    """Compute accuracy of y_true matching top-k probable labels in y_prob.

    :param y_true: ndarray, true label, [n_samples]
    :param y_prob: ndarray, label probabilities, [n_samples, n_classes]
    :param k: int, k in top-k
    :returns acc: accuracy of top-k

    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    y_true_tile = np.tile(np.expand_dims(y_true, axis=1), (1, k))
    y_match = np.any(y_pred_topk == y_true_tile, axis=-1)
    acc = np.sum(y_match) / y_match.shape[0]
    return acc


def _pred_topk(y_prob, k=1):
    """Return top-k predicted labels and corresponding probabilities.

    :param y_prob: ndarray, size [n_samples, n_classes], probabilities on labels
    :param k: int, k of top-k
    :returns (y_pred_topk, y_prob_topk):
        y_pred_topk: ndarray, size [n_samples, k], predicted top-k labels
        y_prob_topk: ndarray, size [n_samples, k], probabilities for top-k labels

    """
    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    x_axis_index = np.tile(
        np.arange(len(y_prob))[:, np.newaxis],
        (1, k))
    y_prob_topk = y_prob[x_axis_index, y_pred_topk]
    return y_pred_topk, y_prob_topk


class CMRC2018Metric(MetricBase):
    def __init__(self, answers=None, raw_chars=None, context_len=None, pred_start=None, pred_end=None):
        super().__init__()
        self._init_param_map(answers=answers, raw_chars=raw_chars, context_len=context_len, pred_start=pred_start,
                             pred_end=pred_end)
        self.em = 0
        self.total = 0
        self.f1 = 0

    def evaluate(self, answers, raw_chars, context_len, pred_start, pred_end):
        """

        :param list[str] answers: 如[["答案1", "答案2", "答案3"], [...], ...]
        :param list[str] raw_chars: [["这", "是", ...], [...]]
        :param tensor context_len: context长度, batch_size
        :param tensor pred_start: batch_size x length
        :param tensor pred_end: batch_size x length
        :return:
        """
        batch_size, max_len = pred_start.size()
        context_mask = seq_len_to_mask(context_len, max_len=max_len).eq(0)
        pred_start.masked_fill_(context_mask, float('-inf'))
        pred_end.masked_fill_(context_mask, float('-inf'))
        max_pred_start, pred_start_index = pred_start.max(dim=-1, keepdim=True)  # batch_size,
        pred_start_mask = pred_start.eq(max_pred_start).cumsum(dim=-1).eq(0)  # 只能预测这之后的值
        pred_end.masked_fill_(pred_start_mask, float('-inf'))
        pred_end_index = pred_end.argmax(dim=-1) + 1
        pred_ans = []
        for index, (start, end) in enumerate(zip(pred_start_index.flatten().tolist(), pred_end_index.tolist())):
            pred_ans.append(''.join(raw_chars[index][start:end]))
        for answer, pred_an in zip(answers, pred_ans):
            pred_an = pred_an.strip()
            self.f1 += _calc_cmrc2018_f1_score(answer, pred_an)
            self.total += 1
            self.em += _calc_cmrc2018_em_score(answer, pred_an)

    def get_metric(self, reset=True):
        eval_res = {'f1': round(self.f1 / self.total*100, 2), 'em': round(self.em / self.total*100, 2)}
        if reset:
            self.em = 0
            self.total = 0
            self.f1 = 0
        return eval_res

# split Chinese
def _cn_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = {'-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：', '？', '！', '“', '”', '；', '’', '《',
               '》', '……', '·', '、', '「', '」', '（', '）', '－', '～', '『', '』'}
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = list(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = list(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def _remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def _find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def _calc_cmrc2018_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _cn_segmentation(ans, rm_punc=True)
        prediction_segs = _cn_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = _find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def _calc_cmrc2018_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _remove_punctuation(ans)
        prediction_ = _remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

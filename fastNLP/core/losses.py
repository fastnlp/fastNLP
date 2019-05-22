"""
losses 模块定义了 fastNLP 中所需的各种损失函数，一般做为 :class:`~fastNLP.Trainer` 的参数使用。

"""
__all__ = [
    "LossBase",
    
    "LossFunc",
    "LossInForward",
    
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",
    "NLLLoss"
]

import inspect
from collections import defaultdict

import torch
import torch.nn.functional as F

from .utils import _CheckError
from .utils import _CheckRes
from .utils import _build_args
from .utils import _check_arg_dict_list
from .utils import _check_function_or_method
from .utils import _get_func_signature


class LossBase(object):
    """
    所有loss的基类。如果想了解其中的原理，请查看源码。
    """
    
    def __init__(self):
        self.param_map = {}
        self._checked = False
    
    def get_loss(self, *args, **kwargs):
        raise NotImplementedError
    
    def _init_param_map(self, key_map=None, **kwargs):
        """检查key_map和其他参数map，并将这些映射关系添加到self.param_map

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
                    self.param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self.param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self.param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self.param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")
        
        # check consistence between signature and param_map
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self.param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature.")
        
        # evaluate should not have varargs.
        # if func_spect.varargs:
        #     raise NameError(f"Delete `*{func_spect.varargs}` in {get_func_signature(self.get_loss)}(Do not use "
        #                     f"positional argument.).")
    
    def _fast_param_map(self, pred_dict, target_dict):
        """Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element

        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not {}, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        if len(self.param_map) == 2 and len(pred_dict) == 1 and len(target_dict) == 1:
            fast_param['pred'] = list(pred_dict.values())[0]
            fast_param['target'] = list(target_dict.values())[0]
            return fast_param
        return fast_param
    
    def __call__(self, pred_dict, target_dict, check=False):
        """
        :param dict pred_dict: 模型的forward函数返回的dict
        :param dict target_dict: DataSet.batch_y里的键-值对所组成的dict
        :param Boolean check: 每一次执行映射函数的时候是否检查映射表，默认为不检查
        :return:
        """
        fast_param = self._fast_param_map(pred_dict, target_dict)
        if fast_param:
            loss = self.get_loss(**fast_param)
            return loss
        
        if not self._checked:
            # 1. check consistence between signature and param_map
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self.param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")
            
            # 2. only part of the param_map are passed, left are not
            for arg in func_args:
                if arg not in self.param_map:
                    self.param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self.param_map.items()}
        
        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        duplicated = []
        for input_arg in set(list(pred_dict.keys()) + list(target_dict.keys())):
            not_duplicate_flag = 0
            if input_arg in self._reverse_param_map:
                mapped_arg = self._reverse_param_map[input_arg]
                not_duplicate_flag += 1
            else:
                mapped_arg = input_arg
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
                not_duplicate_flag += 1
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]
                not_duplicate_flag += 1
            if not_duplicate_flag == 3:
                duplicated.append(input_arg)
        
        # missing
        if not self._checked:
            check_res = _check_arg_dict_list(self.get_loss, [mapped_pred_dict, mapped_target_dict])
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self.param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                    f"in `{self.__class__.__name__}`)"
            
            check_res = _CheckRes(missing=replaced_missing,
                                  unused=check_res.unused,
                                  duplicated=duplicated,
                                  required=check_res.required,
                                  all_needed=check_res.all_needed,
                                  varargs=check_res.varargs)
            
            if check_res.missing or check_res.duplicated:
                raise _CheckError(check_res=check_res,
                                  func_signature=_get_func_signature(self.get_loss))
        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict)
        
        loss = self.get_loss(**refined_args)
        self._checked = True
        
        return loss


class LossFunc(LossBase):
    """
    别名：:class:`fastNLP.LossFunc` :class:`fastNLP.core.losses.LossFunc`

    提供给用户使用自定义损失函数的类

    :param func: 用户自行定义的损失函数，应当为一个函数或者callable(func)为True的ojbect
    :param dict key_map: 参数映射表。键为Model/DataSet参数名，值为损失函数参数名。
                         fastNLP的trainer将在训练时从模型返回值或者训练数据DataSet的target=True的field中
                         找到相对应的参数名为value的参数，并传入func中作为参数名为key的参数
    :param kwargs: 除了参数映射表以外可以用key word args的方式设置参数映射关系

    使用方法::

        func = torch.nn.CrossEntropyLoss()
        loss_func = LossFunc(func, input="pred", target="label")
        # 这表示构建了一个损失函数类，由func计算损失函数，其中将从模型返回值或者DataSet的target=True的field
        # 当中找到一个参数名为`pred`的参数传入func一个参数名为`input`的参数；找到一个参数名为`label`的参数
        # 传入func作为一个名为`target`的参数

    """
    
    def __init__(self, func, key_map=None, **kwargs):
        
        super(LossFunc, self).__init__()
        _check_function_or_method(func)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise RuntimeError(f"Loss error: key_map except a {type({})} but got a {type(key_map)}")
            self.param_map = key_map
        if len(kwargs) > 0:
            for key, val in kwargs.items():
                self.param_map.update({key: val})
        
        self.get_loss = func


class CrossEntropyLoss(LossBase):
    """
    别名：:class:`fastNLP.CrossEntropyLoss` :class:`fastNLP.core.losses.CrossEntropyLoss`

    交叉熵损失函数
    
    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
    :param padding_idx: padding的index，在计算loss时将忽略target中标号为padding_idx的内容

    Example::

        loss = CrossEntropyLoss(pred='pred', target='label', padding_idx=0)
        
    """
    
    def __init__(self, pred=None, target=None, padding_idx=-100):
        # TODO 需要做一些检查，F.cross_entropy在计算时，如果pred是(16, 10 ,4), target的形状按道理应该是(16, 10), 但实际需要（16，4）
        super(CrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        self.padding_idx = padding_idx
    
    def get_loss(self, pred, target):
        return F.cross_entropy(input=pred, target=target,
                               ignore_index=self.padding_idx)


class L1Loss(LossBase):
    """
    别名：:class:`fastNLP.L1Loss` :class:`fastNLP.core.losses.L1Loss`

    L1损失函数
    
    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` >`target`
    
    """
    
    def __init__(self, pred=None, target=None):
        super(L1Loss, self).__init__()
        self._init_param_map(pred=pred, target=target)
    
    def get_loss(self, pred, target):
        return F.l1_loss(input=pred, target=target)


class BCELoss(LossBase):
    """
    别名：:class:`fastNLP.BCELoss` :class:`fastNLP.core.losses.BCELoss`

    二分类交叉熵损失函数
    
    :param pred: 参数映射表中`pred`的映射关系，None表示映射关系为`pred`->`pred`
    :param target: 参数映射表中`target`的映射关系，None表示映射关系为`target`->`target`
    """
    
    def __init__(self, pred=None, target=None):
        super(BCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
    
    def get_loss(self, pred, target):
        return F.binary_cross_entropy(input=pred, target=target)


class NLLLoss(LossBase):
    """
    别名：:class:`fastNLP.NLLLoss` :class:`fastNLP.core.losses.NLLLoss`
    
    负对数似然损失函数
    
    :param pred: 参数映射表中`pred`的映射关系，None表示映射关系为`pred`->`pred`
    :param target: 参数映射表中`target`的映射关系，None表示映射关系为`target`->`target`
    """
    
    def __init__(self, pred=None, target=None):
        super(NLLLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
    
    def get_loss(self, pred, target):
        return F.nll_loss(input=pred, target=target)


class LossInForward(LossBase):
    """
    别名：:class:`fastNLP.LossInForward` :class:`fastNLP.core.losses.LossInForward`

    从forward()函数返回结果中获取loss
    
    :param str loss_key: 在forward函数中loss的键名，默认为loss
    """
    
    def __init__(self, loss_key='loss'):
        super().__init__()
        if not isinstance(loss_key, str):
            raise TypeError(f"Only str allowed for loss_key, got {type(loss_key)}.")
        self.loss_key = loss_key
    
    def get_loss(self, **kwargs):
        if self.loss_key not in kwargs:
            check_res = _CheckRes(
                missing=[self.loss_key + f"(assign to `{self.loss_key}` in `{self.__class__.__name__}`"],
                unused=[],
                duplicated=[],
                required=[],
                all_needed=[],
                varargs=[])
            raise _CheckError(check_res=check_res, func_signature=_get_func_signature(self.get_loss))
        return kwargs[self.loss_key]
    
    def __call__(self, pred_dict, target_dict, check=False):
        
        loss = self.get_loss(**pred_dict)
        
        if not (isinstance(loss, torch.Tensor) and len(loss.size()) == 0):
            if not isinstance(loss, torch.Tensor):
                raise TypeError(f"Loss excepted to be a torch.Tensor, got {type(loss)}")
            loss = torch.sum(loss) / (loss.view(-1)).size(0)
            # raise RuntimeError(f"The size of loss excepts to be torch.Size([]), got {loss.size()}")
        
        return loss


def _prepare_losser(losser):
    if losser is None:
        losser = LossInForward()
        return losser
    elif isinstance(losser, LossBase):
        return losser
    else:
        raise TypeError(f"Type of loss should be `fastNLP.LossBase`, got {type(losser)}")


def squash(predict, truth, **kwargs):
    """To reshape tensors in order to fit loss functions in PyTorch.

    :param predict: Tensor, model output
    :param truth: Tensor, truth from dataset
    :param kwargs: extra arguments
    :return predict , truth: predict & truth after processing
    """
    return predict.view(-1, predict.size()[-1]), truth.view(-1, )


def unpad(predict, truth, **kwargs):
    """To process padded sequence output to get true loss.

    :param predict: Tensor, [batch_size , max_len , tag_size]
    :param truth: Tensor, [batch_size , max_len]
    :param kwargs: kwargs["lens"] is a list or LongTensor, with size [batch_size]. The i-th element is true lengths of i-th sequence.

    :return predict , truth: predict & truth after processing
    """
    if kwargs.get("lens") is None:
        return predict, truth
    lens = torch.LongTensor(kwargs["lens"])
    lens, idx = torch.sort(lens, descending=True)
    predict = torch.nn.utils.rnn.pack_padded_sequence(predict[idx], lens, batch_first=True).data
    truth = torch.nn.utils.rnn.pack_padded_sequence(truth[idx], lens, batch_first=True).data
    return predict, truth


def unpad_mask(predict, truth, **kwargs):
    """To process padded sequence output to get true loss.

    :param predict: Tensor, [batch_size , max_len , tag_size]
    :param truth: Tensor, [batch_size , max_len]
    :param kwargs: kwargs["lens"] is a list or LongTensor, with size [batch_size]. The i-th element is true lengths of i-th sequence.

    :return predict , truth: predict & truth after processing
    """
    if kwargs.get("lens") is None:
        return predict, truth
    mas = make_mask(kwargs["lens"], truth.size()[1])
    return mask(predict, truth, mask=mas)


def mask(predict, truth, **kwargs):
    """To select specific elements from Tensor. This method calls ``squash()``.

    :param predict: Tensor, [batch_size , max_len , tag_size]
    :param truth: Tensor, [batch_size , max_len]
    :param kwargs: extra arguments, kwargs["mask"]: ByteTensor, [batch_size , max_len], the mask Tensor. The position that is 1 will be selected.

    :return predict , truth: predict & truth after processing
    """
    if kwargs.get("mask") is None:
        return predict, truth
    mask = kwargs["mask"]
    
    predict, truth = squash(predict, truth)
    mask = mask.view(-1, )
    
    predict = torch.masked_select(predict.permute(1, 0), mask).view(predict.size()[-1], -1).permute(1, 0)
    truth = torch.masked_select(truth, mask)
    
    return predict, truth


def make_mask(lens, tar_len):
    """To generate a mask over a sequence.

    :param lens: list or LongTensor, [batch_size]
    :param tar_len: int
    :return mask: ByteTensor
    """
    lens = torch.LongTensor(lens)
    mask = [torch.ge(lens, i + 1) for i in range(tar_len)]
    mask = torch.stack(mask, 1)
    return mask

r"""
losses 模块定义了 fastNLP 中所需的各种损失函数，一般做为 :class:`~fastNLP.Trainer` 的参数使用。

"""
__all__ = [
    "LossBase",
    
    "LossFunc",
    "LossInForward",
    
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",
    "NLLLoss",
    "MSELoss",

    "CMRC2018Loss"

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
from .utils import seq_len_to_mask
from ..core.const import Const


class LossBase(object):
    r"""
    所有loss的基类。如果需要结合到Trainer之中需要实现get_loss方法
    """
    
    def __init__(self):
        self._param_map = {}  # key是fun的参数，value是以该值从传入的dict取出value
        self._checked = False

    @property
    def param_map(self):
        if len(self._param_map) == 0:  # 如果为空说明还没有初始化
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = [arg for arg in func_spect.args if arg != 'self']
            for arg in func_args:
                self._param_map[arg] = arg
        return self._param_map

    def get_loss(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return: torch.Tensor
        """
        raise NotImplementedError
    
    def _init_param_map(self, key_map=None, **kwargs):
        r"""检查key_map和其他参数map，并将这些映射关系添加到self._param_map

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
        func_spect = inspect.getfullargspec(self.get_loss)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self._param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {_get_func_signature(self.get_loss)}. Please check the "
                    f"initialization parameters, or change its signature.")
        
        # evaluate should not have varargs.
        # if func_spect.varargs:
        #     raise NameError(f"Delete `*{func_spect.varargs}` in {get_func_signature(self.get_loss)}(Do not use "
        #                     f"positional argument.).")

    def __call__(self, pred_dict, target_dict, check=False):
        r"""
        :param dict pred_dict: 模型的forward函数返回的dict
        :param dict target_dict: DataSet.batch_y里的键-值对所组成的dict
        :param Boolean check: 每一次执行映射函数的时候是否检查映射表，默认为不检查
        :return:
        """

        if not self._checked:
            # 1. check consistence between signature and _param_map
            func_spect = inspect.getfullargspec(self.get_loss)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self._param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {_get_func_signature(self.get_loss)}.")
            
            # 2. only part of the _param_map are passed, left are not
            for arg in func_args:
                if arg not in self._param_map:
                    self._param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self._param_map.items()}

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
            check_res = _check_arg_dict_list(self.get_loss, [mapped_pred_dict, mapped_target_dict])
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
                                  func_signature=_get_func_signature(self.get_loss))
            self._checked = True

        refined_args = _build_args(self.get_loss, **mapped_pred_dict, **mapped_target_dict)
        
        loss = self.get_loss(**refined_args)
        self._checked = True
        
        return loss


class LossFunc(LossBase):
    r"""
    提供给用户使用自定义损失函数的类

    :param func: 用户自行定义的损失函数，应当为一个函数。
    :param dict key_map: 参数映射表。键为Model/DataSet参数名，值为损失函数参数名。
                         fastNLP的trainer将在训练时从模型返回值或者训练数据DataSet的target=True的field中
                         找到相对应的参数名为value的参数，并传入func中作为参数名为key的参数
    :param kwargs: 除了参数映射表以外可以用key word args的方式设置参数映射关系

    使用方法::

        import torch.nn.functional as F
        loss_func = LossFunc(F.cross_entropy, input="pred", target="label")
        # 这表示构建了一个损失函数类，由func计算损失函数，其中将从模型返回值或者DataSet的target=True的field
        # 当中找到一个参数名为`pred`的参数传入func一个参数名为`input`的参数；找到一个参数名为`label`的参数
        # 传入func作为一个名为`target`的参数

    """
    
    def __init__(self, func, key_map=None, **kwargs):
        
        super(LossFunc, self).__init__()
        _check_function_or_method(func)
        self.get_loss = func
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise RuntimeError(f"Loss error: key_map except a {type({})} but got a {type(key_map)}")
        self._init_param_map(key_map, **kwargs)


class CrossEntropyLoss(LossBase):
    r"""
    交叉熵损失函数
    
    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
    :param seq_len: 句子的长度, 长度之外的token不会计算loss。
    :param int class_in_dim: 在序列标注的场景中，pred可能的shape为(batch_size, max_len, num_classes)
        或(batch_size, num_classes, max_len)， CrossEntropyLoss需要知道哪一维是class的维度以计算loss。如果为-1，就根据pred的第
        二维是否等于target的第二维来判断是否需要交换pred的第二维和第三维，因为target的第二维是length的维度，如果这一维度上和pred相等，
        那么pred可能第二维也是长度维(存在误判的可能，如果有误判的情况，请显示设置该值)。其它大于0的值则认为该维度是class的维度。
    :param padding_idx: padding的index，在计算loss时将忽略target中标号为padding_idx的内容, 可以通过该值代替
        传入seq_len.
    :param str reduction: 支持 `mean` ，`sum` 和 `none` .

    Example::

        loss = CrossEntropyLoss(pred='pred', target='label', padding_idx=0)
        
    """
    
    def __init__(self, pred=None, target=None, seq_len=None, class_in_dim=-1, padding_idx=-100, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.padding_idx = padding_idx
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.class_in_dim = class_in_dim
    
    def get_loss(self, pred, target, seq_len=None):
        if seq_len is not None and target.dim()>1:
            mask = seq_len_to_mask(seq_len, max_len=target.size(1)).eq(False)
            target = target.masked_fill(mask, self.padding_idx)

        if pred.dim() > 2:
            if self.class_in_dim == -1:
                if pred.size(1) != target.size(1):  # 有可能顺序替换了
                    pred = pred.transpose(1, 2)
            else:
                pred = pred.transpose(-1, self.class_in_dim)
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)

        return F.cross_entropy(input=pred, target=target,
                               ignore_index=self.padding_idx, reduction=self.reduction)


class L1Loss(LossBase):
    r"""
    L1损失函数
    
    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` >`target`
    :param str reduction: 支持'mean'，'sum'和'none'.
    
    """
    
    def __init__(self, pred=None, target=None, reduction='mean'):
        super(L1Loss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    
    def get_loss(self, pred, target):
        return F.l1_loss(input=pred, target=target, reduction=self.reduction)


class MSELoss(LossBase):
    r"""
    MSE损失函数

    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` >`target`
    :param str reduction: 支持'mean'，'sum'和'none'.

    """

    def __init__(self, pred=None, target=None, reduction='mean'):
        super(MSELoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction

    def get_loss(self, pred, target):
        return F.mse_loss(input=pred, target=target, reduction=self.reduction)


class BCELoss(LossBase):
    r"""
    二分类交叉熵损失函数
    
    :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
    :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
    :param str reduction: 支持 `mean` ，`sum` 和 `none` .
    """
    
    def __init__(self, pred=None, target=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
    
    def get_loss(self, pred, target):
        return F.binary_cross_entropy(input=pred, target=target, reduction=self.reduction)


class NLLLoss(LossBase):
    r"""
    负对数似然损失函数
    """
    
    def __init__(self, pred=None, target=None, ignore_idx=-100, reduction='mean'):
        r"""
        
        :param pred: 参数映射表中 `pred` 的映射关系，None表示映射关系为 `pred` -> `pred`
        :param target: 参数映射表中 `target` 的映射关系，None表示映射关系为 `target` -> `target`
        :param ignore_idx: ignore的index，在计算loss时将忽略target中标号为ignore_idx的内容, 可以通过该值代替
            传入seq_len.
        :param str reduction: 支持 `mean` ，`sum` 和 `none` .
        """
        super(NLLLoss, self).__init__()
        self._init_param_map(pred=pred, target=target)
        assert reduction in ('mean', 'sum', 'none')
        self.reduction = reduction
        self.ignore_idx = ignore_idx
    
    def get_loss(self, pred, target):
        return F.nll_loss(input=pred, target=target, ignore_index=self.ignore_idx, reduction=self.reduction)


class LossInForward(LossBase):
    r"""
    从forward()函数返回结果中获取loss
    """
    
    def __init__(self, loss_key=Const.LOSS):
        r"""
        
        :param str loss_key: 在forward函数中loss的键名，默认为loss
        """
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


class CMRC2018Loss(LossBase):
    r"""
    用于计算CMRC2018中文问答任务。

    """
    def __init__(self, target_start=None, target_end=None, context_len=None, pred_start=None, pred_end=None,
                  reduction='mean'):
        super().__init__()

        assert reduction in ('mean', 'sum')

        self._init_param_map(target_start=target_start, target_end=target_end, context_len=context_len,
                             pred_start=pred_start, pred_end=pred_end)
        self.reduction = reduction

    def get_loss(self, target_start, target_end, context_len, pred_start, pred_end):
        r"""

        :param target_start: batch_size
        :param target_end: batch_size
        :param context_len: batch_size
        :param pred_start: batch_size x max_len
        :param pred_end: batch_size x max_len
        :return:
        """
        batch_size, max_len = pred_end.size()
        mask = seq_len_to_mask(context_len, max_len).eq(False)

        pred_start = pred_start.masked_fill(mask, float('-inf'))
        pred_end = pred_end.masked_fill(mask, float('-inf'))

        start_loss = F.cross_entropy(pred_start, target_start, reduction='sum')
        end_loss = F.cross_entropy(pred_end, target_end, reduction='sum')

        loss = start_loss + end_loss

        if self.reduction == 'mean':
            loss = loss / batch_size

        return loss/2

def _prepare_losser(losser):
    if losser is None:
        losser = LossInForward()
        return losser
    elif isinstance(losser, LossBase):
        return losser
    else:
        raise TypeError(f"Type of loss should be `fastNLP.LossBase`, got {type(losser)}")

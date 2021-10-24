r"""
utils模块实现了 fastNLP 内部和外部所需的很多工具。其中用户可以使用的是 :func:`cache_results` 修饰器。
"""

__all__ = [
    "cache_results",
    "seq_len_to_mask",
    "get_seq_len"
]

import inspect
import os
import warnings
from collections import Counter, namedtuple
from typing import List

import _pickle
import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from ._logger import logger
from ._parallel_utils import _model_contains_inner_module
# from .vocabulary import Vocabulary
import torch
import contextlib
from pkg_resources import parse_version


_CheckRes = namedtuple('_CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                     'varargs'])


class ConfusionMatrix:
    r"""a dict can provide Confusion Matrix"""
    def __init__(self, show_result=None,vocab=None, print_ratio=False):
        r"""
        :param show_result: list type, 数据类型需要和target保持一致
        :param vocab: 需要有to_word方法，建议直接使用Fastnlp.core.Vocabulary。
        :param print_ratio: 限制print的输出，False只输出数量Confusion Matrix, True还会输出百分比Confusion Matrix, 分别为行/列
        """
        if vocab and not hasattr(vocab, "to_word"):
            raise TypeError(
                f"`vocab` in {_get_func_signature(self.__init__)} must be Fastnlp.core.Vocabulary,"
                f"got {type(vocab)}.")
        self.confusiondict = {}  # key: pred index, value:target word ocunt
        self.predcount = {}  # key:pred index, value:count
        self.targetcount = {}  # key:target index, value:count
        self.show_result = show_result
        self.vocab = vocab
        self.print_ratio = print_ratio

    def add_pred_target(self, pred, target):  # 一组结果
        r"""
        通过这个函数向ConfusionMatrix加入一组预测结果
        :param list pred: 预测的标签列表
        :param list target: 真实值的标签列表
        :return ConfusionMatrix
        confusion=ConfusionMatrix()
        pred = [2,1,3]
        target = [2,2,1]
        confusion.add_pred_target(pred, target)
        print(confusion)

        target  1       2       3       all
          pred
             1  0       1       0         1
             2  0       1       0         1
             3  1       0       0         1
           all  1       2       0         3
        """
        for p, t in zip(pred, target):  # <int, int>
            self.predcount[p] = self.predcount.get(p, 0) + 1
            self.targetcount[t] = self.targetcount.get(t, 0) + 1
            if p in self.confusiondict:
                self.confusiondict[p][t] = self.confusiondict[p].get(t, 0) + 1
            else:
                self.confusiondict[p] = {}
                self.confusiondict[p][t] = 1
        return self.confusiondict

    def clear(self):
        r"""
        清空ConfusionMatrix，等待再次新加入
        :return: 
        """
        self.confusiondict = {}
        self.targetcount = {}
        self.predcount = {}

    def get_result(self):
        r"""
        :return list output: ConfusionMatrix content,具体值与汇总统计
        """
        row2idx = {}
        idx2row = {}
        # 已知的所有键/label
        totallabel = sorted(
            list(
                set(self.targetcount.keys()).union(set(
                    self.predcount.keys()))))
        lenth = len(totallabel)

        for label, idx in zip(totallabel, range(lenth)):
            idx2row[
                label] = idx  # 建立一个临时字典，key:vocab的index, value: 行列index  1,3,5...->0,1,2,...
            row2idx[
                idx] = label  # 建立一个临时字典，value:vocab的index, key: 行列index  0,1,2...->1,3,5,...
        output = []
        for i in row2idx.keys():  # 第i行
            p = row2idx[i]
            l = [0 for _ in range(lenth)]
            if self.confusiondict.get(p, None):
                for t, c in self.confusiondict[p].items():
                    l[idx2row[t]] = c  # 完成一行
            l = [n for n in l] + [sum(l)]
            output.append(l)
        tail = [self.targetcount.get(row2idx[k], 0) for k in row2idx.keys()]
        tail += [sum(tail)]
        output.append(tail)
        return output

    def get_percent(self, dim=0):
        r"""
        :param dim int: 0/1, 0 for row,1 for column
        :return list output: ConfusionMatrix content,具体值与汇总统计
        """
        result = self.get_result()
        if dim == 0:
            tmp = np.array(result)
            tmp = tmp / (tmp[:, -1].reshape([len(result), -1]))
            tmp[np.isnan(tmp)] = 0
            tmp = tmp * 100
        elif dim == 1:
            tmp = np.array(result).T
            tmp = tmp / (tmp[:, -1].reshape([len(result), -1]) + 1e-12)
            tmp = tmp.T * 100
        tmp = np.around(tmp, decimals=2)
        return tmp.tolist()

    def get_aligned_table(self, data, flag="result"):
        r"""
        :param data: highly recommend use get_percent/ get_result return as dataset here, or make sure data is a n*n list type data
        :param flag: only difference between result and other words is whether "%" is in output string
        :return: an aligned_table ready to print out
        """
        row2idx = {}
        idx2row = {}
        # 已知的所有键/label
        totallabel = sorted(
            list(
                set(self.targetcount.keys()).union(set(
                    self.predcount.keys()))))
        lenth = len(totallabel)
        # namedict key :label idx value: str label name/label idx
        namedict = dict([
            (k, str(k if self.vocab == None else self.vocab.to_word(k)))
            for k in totallabel
        ])
        for label, lineidx in zip(totallabel, range(lenth)):
            idx2row[
                label] = lineidx  # 建立一个临时字典，key:vocab的index, value: 行列index  1,3,5...->0,1,2,...
            row2idx[
                lineidx] = label  # 建立一个临时字典，key: 行列index  0,1,2...->1,3,5,...,value:vocab的index,
        # 这里打印东西
        out = str()
        output = []
        # 表头
        head = (["target"] +
                [str(namedict[row2idx[k]]) for k in row2idx.keys()] + ["all"])
        col_lenths = [len(h) for h in head]
        output.append(head)
        output.append(["pred"])
        # 内容
        for i in row2idx.keys():  # 第i行
            p = row2idx[i]
            h = namedict[p]
            l = [h] + [[str(n) + "%", str(n)][flag == "result"]
                       for n in data[i]]
            col_lenths = [
                max(col_lenths[idx], [len(i) for i in l][idx])
                for idx in range(len(col_lenths))
            ]
            output.append(l)

        tail = ["all"] + [[str(n) + "%", str(n)][flag == "result"]
                          for n in data[-1]]
        col_lenths = [
            max(col_lenths[idx], [len(i) for i in tail][idx])
            for idx in range(len(col_lenths))
        ]
        output.append(tail)

        if self.show_result:
            missing_item=[]
            missing_item = [i for i in self.show_result if i not in idx2row]
            self.show_result = [i for i in self.show_result if i in idx2row]
            if missing_item:
                print(f"Noticing label(s) which is/are not in target list appeared, final output string will not contain{str(missing_item)}")
            if self.show_result:
                show_col = [0] + [i + 1 for i in [idx2row[i] for i in self.show_result]]
                show_row = [0]+[i+2 for i in [idx2row[i] for i in self.show_result]]
                output = [[row[col] for col in show_col] for row in [output[row] for row in show_row]]
                output.insert(1,["pred"])
        for line in output:
            for colidx in range(len(line)):
                out += "%*s" % (col_lenths[colidx], line[colidx]) + "\t"
            out += "\n"
        return "\n" + out

    def __repr__(self):
        r"""
        :return string output: ConfusionMatrix的格式化输出，包括表头各标签字段，具体值与汇总统计。
        """
        result = self.get_result()
        o0 = self.get_aligned_table(result, flag="result")

        out = str()
        if self.print_ratio:
            p1 = self.get_percent()
            o1 = "\nNotice the row direction\n" + self.get_aligned_table(
                p1, flag="percent")
            p2 = self.get_percent(dim=1)
            o2 = "\nNotice the column direction\n" + self.get_aligned_table(
                p2, flag="percent")
            out = out + o0 + o1 + o2
        else:
            out = o0
        return out



class Option(dict):
    r"""a dict can treat keys as attributes"""

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        if key.startswith('__') and key.endswith('__'):
            raise AttributeError(key)
        self.__setitem__(key, value)

    def __delattr__(self, item):
        try:
            self.pop(item)
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)


def _prepare_cache_filepath(filepath):
    r"""
    检查filepath是否可以作为合理的cache文件. 如果可以的话，会自动创造路径
    :param filepath: str.
    :return: None, if not, this function will raise error
    """
    _cache_filepath = os.path.abspath(filepath)
    if os.path.isdir(_cache_filepath):
        raise RuntimeError("The cache_file_path must be a file, not a directory.")
    cache_dir = os.path.dirname(_cache_filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)


def cache_results(_cache_fp, _refresh=False, _verbose=1):
    r"""
    cache_results是fastNLP中用于cache数据的装饰器。通过下面的例子看一下如何使用::

        import time
        import numpy as np
        from fastNLP import cache_results
        
        @cache_results('cache.pkl')
        def process_data():
            # 一些比较耗时的工作，比如读取数据，预处理数据等，这里用time.sleep()代替耗时
            time.sleep(1)
            return np.random.randint(10, size=(5,))
        
        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)
        
        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)
        
        # 输出内容如下，可以看到两次结果相同，且第二次几乎没有花费时间
        # Save cache to cache.pkl.
        # res = [5 4 9 1 8]
        # 1.0042750835418701
        # Read cache from cache.pkl.
        # res = [5 4 9 1 8]
        # 0.0040721893310546875

    可以看到第二次运行的时候，只用了0.0001s左右，是由于第二次运行将直接从cache.pkl这个文件读取数据，而不会经过再次预处理::

        # 还是以上面的例子为例，如果需要重新生成另一个cache，比如另一个数据集的内容，通过如下的方式调用即可
        process_data(_cache_fp='cache2.pkl')  # 完全不影响之前的‘cache.pkl'

    上面的_cache_fp是cache_results会识别的参数，它将从'cache2.pkl'这里缓存/读取数据，即这里的'cache2.pkl'覆盖默认的
    'cache.pkl'。如果在你的函数前面加上了@cache_results()则你的函数会增加三个参数[_cache_fp, _refresh, _verbose]。
    上面的例子即为使用_cache_fp的情况，这三个参数不会传入到你的函数中，当然你写的函数参数名也不可能包含这三个名称::

        process_data(_cache_fp='cache2.pkl', _refresh=True)  # 这里强制重新生成一份对预处理的cache。
        #  _verbose是用于控制输出信息的，如果为0,则不输出任何内容;如果为1,则会提醒当前步骤是读取的cache还是生成了新的cache

    :param str _cache_fp: 将返回结果缓存到什么位置;或从什么位置读取缓存。如果为None，cache_results没有任何效用，除非在
        函数调用的时候传入_cache_fp这个参数。
    :param bool _refresh: 是否重新生成cache。
    :param int _verbose: 是否打印cache的信息。
    :return:
    """

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = _pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    _prepare_cache_filepath(cache_filepath)
                    with open(cache_filepath, 'wb') as f:
                        _pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_


def _save_model(model, model_name, save_dir, only_param=False):
    r""" 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param save_dir: 保存的directory
    :param only_param:
    :return:
    """
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if _model_contains_inner_module(model):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
    else:
        _model_device = _get_model_device(model)
        model.cpu()
        torch.save(model, model_path)
        model.to(_model_device)


def _move_model_to_device(model, device):
    r"""
    将model移动到device

    :param model: torch.nn.DataParallel or torch.nn.Module. 当为torch.nn.DataParallel, 则只是调用一次cuda。device必须为
        None。
    :param str,int,torch.device,list(int),list(torch.device) device: 将模型load到哪个设备。默认为None，即Trainer不对模型
        的计算位置进行管理。支持以下的输入:

        1. str: ['cpu', 'cuda', 'cuda:0', 'cuda:1', ...] 依次为'cpu'中, 可见的第一个GPU中, 可见的第一个GPU中,
        可见的第二个GPU中;

        2. torch.device：将模型装载到torch.device上。

        3. int: 将使用device_id为该值的gpu进行训练

        4. list(int)：如果多于1个device，将使用torch.nn.DataParallel包裹model, 并使用传入的device。

        5. None. 为None则不对模型进行任何处理，如果传入的model为torch.nn.DataParallel该值必须为None。

    :return: torch.nn.DataParallel or torch.nn.Module
    """
    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     raise RuntimeError("model of `torch.nn.parallel.DistributedDataParallel` is not supported right now.")

    if device is None:
        if isinstance(model, torch.nn.DataParallel):
            model.cuda(model.device_ids[0])
        return model
    else:
        if not torch.cuda.is_available() and ((isinstance(device, str) and device!='cpu') or
         (isinstance(device, torch.device) and device.type != 'cpu')):
            raise ValueError("There is no usable gpu. set `device` as `cpu` or `None`.")

    if isinstance(model, torch.nn.DataParallel):
        raise RuntimeError("When model is `torch.nn.DataParallel`, the device has to be `None`.")

    if isinstance(device, int):
        assert device > -1, "device can only be non-negative integer"
        assert torch.cuda.device_count() > device, "Only has {} gpus, cannot use device {}.".format(
            torch.cuda.device_count(),
            device)
        device = torch.device('cuda:{}'.format(device))
    elif isinstance(device, str):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, torch.device):
        if device.type == 'cuda' and device.index is not None:
            assert device.index < torch.cuda.device_count(), "Only has {} gpus, cannot use device cuda:{}.".format(
                torch.cuda.device_count(),
                device)
    elif isinstance(device, list):
        types = set([type(d) for d in device])
        assert len(types) == 1, "Mixed type in device, only `int` allowed."
        assert list(types)[0] == int, "Only int supported for multiple devices."
        assert len(set(device)) == len(device), "Duplicated device id found in device."
        for d in device:
            assert d > -1, "Only non-negative device id allowed."
        if len(device) > 1:
            output_device = device[0]
            model = nn.DataParallel(model, device_ids=device, output_device=output_device)
        device = torch.device(device[0])
    else:
        raise TypeError("Unsupported device type.")
    model = model.to(device)
    return model


def _get_model_device(model):
    r"""
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    # TODO 这个函数存在一定的风险，因为同一个模型可能存在某些parameter不在显卡中，比如BertEmbedding. 或者跨显卡
    assert isinstance(model, nn.Module)

    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _build_args(func, **kwargs):
    r"""
    根据func的初始化参数，从kwargs中选择func需要的参数

    :param func: callable
    :param kwargs: 参数
    :return:dict. func中用到的参数
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output


def _map_args(maps: dict, **kwargs):
    # maps: key=old name, value= new name
    output = {}
    for name, val in kwargs.items():
        if name in maps:
            assert isinstance(maps[name], str)
            output.update({maps[name]: val})
        else:
            output.update({name: val})
    for keys in maps.keys():
        if keys not in output.keys():
            pass
    return output


def _get_arg_list(func):
    assert callable(func)
    spect = inspect.getfullargspec(func)
    if spect.defaults is not None:
        args = spect.args[: -len(spect.defaults)]
        defaults = spect.args[-len(spect.defaults):]
        defaults_val = spect.defaults
    else:
        args = spect.args
        defaults = None
        defaults_val = None
    varargs = spect.varargs
    kwargs = spect.varkw
    return args, defaults, defaults_val, varargs, kwargs


# check args
def _check_arg_dict_list(func, args):
    if isinstance(args, dict):
        arg_dict_list = [args]
    else:
        arg_dict_list = args
    assert callable(func) and isinstance(arg_dict_list, (list, tuple))
    assert len(arg_dict_list) > 0 and isinstance(arg_dict_list[0], dict)
    spect = inspect.getfullargspec(func)
    all_args = set([arg for arg in spect.args if arg != 'self'])
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    default_args = set(spect.args[start_idx:])
    require_args = all_args - default_args
    input_arg_count = Counter()
    for arg_dict in arg_dict_list:
        input_arg_count.update(arg_dict.keys())
    duplicated = [name for name, val in input_arg_count.items() if val > 1]
    input_args = set(input_arg_count.keys())
    missing = list(require_args - input_args)
    unused = list(input_args - all_args)
    varargs = [] if not spect.varargs else [spect.varargs]
    return _CheckRes(missing=missing,
                     unused=unused,
                     duplicated=duplicated,
                     required=list(require_args),
                     all_needed=list(all_args),
                     varargs=varargs)


def _get_func_signature(func):
    r"""

    Given a function or method, return its signature.
    For example:
    
    1 function::
    
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
        
    2 method::
    
        class Demo:
            def __init__(self):
                xxx
            def forward(self, a, b='a', **args)
        demo = Demo()
        get_func_signature(demo.forward) # 'Demo.forward(self, a, b='a', **args)'
        
    :param func: a function or a method
    :return: str or None
    """
    if inspect.ismethod(func):
        class_name = func.__self__.__class__.__name__
        signature = inspect.signature(func)
        signature_str = str(signature)
        if len(signature_str) > 2:
            _self = '(self, '
        else:
            _self = '(self'
        signature_str = class_name + '.' + func.__name__ + _self + signature_str[1:]
        return signature_str
    elif inspect.isfunction(func):
        signature = inspect.signature(func)
        signature_str = str(signature)
        signature_str = func.__name__ + signature_str
        return signature_str


def _is_function_or_method(func):
    r"""

    :param func:
    :return:
    """
    if not inspect.ismethod(func) and not inspect.isfunction(func):
        return False
    return True


def _check_function_or_method(func):
    if not _is_function_or_method(func):
        raise TypeError(f"{type(func)} is not a method or function.")


def _move_dict_value_to_device(*args, device: torch.device, non_blocking=False):
    r"""

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param non_blocking: bool, 是否异步将数据转移到cpu, 需要tensor使用pin_memory()
    :param args:
    :return:
    """
    if not torch.cuda.is_available() or device is None:
        return

    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device, non_blocking=non_blocking)
        else:
            raise TypeError("Only support `dict` type right now.")


class _CheckError(Exception):
    r"""

    _CheckError. Used in losses.LossBase, metrics.MetricBase.
    """

    def __init__(self, check_res: _CheckRes, func_signature: str):
        errs = [f'Problems occurred when calling `{func_signature}`']

        if check_res.varargs:
            errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
        if check_res.missing:
            errs.append(f"\tmissing param: {check_res.missing}")
        if check_res.duplicated:
            errs.append(f"\tduplicated param: {check_res.duplicated}")
        if check_res.unused:
            errs.append(f"\tunused param: {check_res.unused}")

        Exception.__init__(self, '\n'.join(errs))

        self.check_res = check_res
        self.func_signature = func_signature


IGNORE_CHECK_LEVEL = 0
WARNING_CHECK_LEVEL = 1
STRICT_CHECK_LEVEL = 2


def _check_loss_evaluate(prev_func_signature: str, func_signature: str, check_res: _CheckRes,
                         pred_dict: dict, target_dict: dict, dataset, check_level=0):
    errs = []
    unuseds = []
    _unused_field = []
    _unused_param = []
    suggestions = []
    # if check_res.varargs:
    #     errs.append(f"\tvarargs: *{check_res.varargs}")
    #     suggestions.append(f"Does not support pass positional arguments, please delete *{check_res.varargs}.")

    if check_res.unused:
        for _unused in check_res.unused:
            if _unused in target_dict:
                _unused_field.append(_unused)
            else:
                _unused_param.append(_unused)
        if _unused_field:
            unuseds.append(f"\tunused field: {_unused_field}")
        if _unused_param:
            unuseds.append(f"\tunused param: {_unused_param}")  # output from predict or forward

    module_name = func_signature.split('.')[0]
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        import re
        mapped_missing = []  # 提供了映射的参数
        unmapped_missing = []  # 没有指定映射的参数
        input_func_map = {}
        for _miss_ in check_res.missing:
            # they shoudl like 'SomeParam(assign to xxx)'
            _miss = _miss_.split('(')[0]
            matches = re.findall("(?<=`)[a-zA-Z0-9]*?(?=`)", _miss_)
            if len(matches) == 2:
                fun_arg, module_name = matches
                input_func_map[_miss] = fun_arg
                if fun_arg == _miss:
                    unmapped_missing.append(_miss)
                else:
                    mapped_missing.append(_miss)
            else:
                unmapped_missing.append(_miss)

        for _miss in mapped_missing + unmapped_missing:
            if _miss in dataset:
                suggestions.append(f"Set `{_miss}` as target.")
            else:
                _tmp = ''
                if check_res.unused:
                    _tmp = f"Check key assignment for `{input_func_map.get(_miss,_miss)}` when initialize {module_name}."
                if _tmp:
                    _tmp += f' Or provide `{_miss}` in DataSet or the output of {prev_func_signature}. '
                else:
                    _tmp = f'Provide `{_miss}` in DataSet or the output of {prev_func_signature}.'
                if not dataset.collater.is_empty():
                    _tmp += f'Or you need to add `{_miss}` in the output of your collate_fn. '
                suggestions.append(_tmp)

    if check_res.duplicated:
        errs.append(f"\tduplicated param: {check_res.duplicated}.")
        suggestions.append(f"Delete {check_res.duplicated} in the output of "
                           f"{prev_func_signature} or do not set {check_res.duplicated} as targets. ")

    if len(errs) > 0:
        errs.extend(unuseds)
    elif check_level == STRICT_CHECK_LEVEL:
        errs.extend(unuseds)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                if idx > 0:
                    sugg_str += '\t\t\t'
                sugg_str += f'({idx + 1}). {sugg}\n'
            sugg_str = sugg_str[:-1]
        else:
            sugg_str += suggestions[0]
        errs.append(f'\ttarget field: {list(target_dict.keys())}')
        errs.append(f'\tparam from {prev_func_signature}: {list(pred_dict.keys())}')
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if check_res.unused:
        if check_level == WARNING_CHECK_LEVEL:
            if not module_name:
                module_name = func_signature.split('.')[0]
            _unused_warn = f'{check_res.unused} is not used by {module_name}.'
            warnings.warn(message=_unused_warn)


def _check_forward_error(forward_func, batch_x, dataset, check_level):
    check_res = _check_arg_dict_list(forward_func, batch_x)
    func_signature = _get_func_signature(forward_func)

    errs = []
    suggestions = []
    _unused = []

    # if check_res.varargs:
    #     errs.append(f"\tvarargs: {check_res.varargs}")
    #     suggestions.append(f"Does not support pass positional arguments, please delete *{check_res.varargs}.")
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        _miss_in_dataset = []
        _miss_out_dataset = []
        for _miss in check_res.missing:
            if _miss in dataset:
                _miss_in_dataset.append(_miss)
            else:
                _miss_out_dataset.append(_miss)
        if _miss_in_dataset:
            suggestions.append(f"You might need to set `{_miss_in_dataset}` as input. ")
        if _miss_out_dataset:
            _tmp = f"You need to provide `{_miss_out_dataset}` in DataSet and set it as input. "
            if not dataset.collater.is_empty():
                _tmp += f'Or you need to add `{_miss_out_dataset}` in the output of your collate_fn. '
            suggestions.append(_tmp)

    if check_res.unused:
        _unused = [f"\tunused field: {check_res.unused}"]
        if len(errs) > 0:
            errs.extend(_unused)
        elif check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                sugg_str += f'({idx + 1}). {sugg}'
            err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        elif len(suggestions):
            sugg_str += suggestions[0]
            err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        else:
            err_str = '\n' + '\n'.join(errs)
        raise NameError(err_str)
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def seq_len_to_mask(seq_len, max_len=None):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


class _pseudo_tqdm:
    r"""
    当无法引入tqdm，或者Trainer中设置use_tqdm为false的时候，用该方法打印数据
    """

    def __init__(self, **kwargs):
        self.logger = logger

    def write(self, info):
        self.logger.info(info)

    def set_postfix_str(self, info):
        self.logger.info(info)

    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass

        return pass_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self


def iob2(tags: List[str]) -> List[str]:
    r"""
    检查数据是否是合法的IOB数据，如果是IOB1会被自动转换为IOB2。两者的差异见
        https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format

    :param tags: 需要转换的tags, 需要为大写的BIO标签。
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return tags


def iob2bioes(tags: List[str]) -> List[str]:
    r"""
    将iob的tag转换为bioes编码
    :param tags: List[str]. 编码需要是大写的。
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')[0]
            if split == 'B':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise TypeError("Invalid IOB format.")
    return new_tags


def _is_iterable(value):
    # 检查是否是iterable的, duck typing
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


def get_seq_len(words, pad_value=0):
    r"""
    给定batch_size x max_len的words矩阵，返回句子长度

    :param words: batch_size x max_len
    :return: (batch_size,)
    """
    mask = words.ne(pad_value)
    return mask.sum(dim=-1)


def pretty_table_printer(dataset_or_ins) -> PrettyTable:
    r"""
    :param dataset_or_ins: 传入一个dataSet或者instance
    ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2], field_3=["a", "b", "c"])
    +-----------+-----------+-----------------+
    |  field_1  |  field_2  |     field_3     |
    +-----------+-----------+-----------------+
    | [1, 1, 1] | [2, 2, 2] | ['a', 'b', 'c'] |
    +-----------+-----------+-----------------+
    :return: 以 pretty table的形式返回根据terminal大小进行自动截断
    """
    x = PrettyTable()
    try:
        sz = os.get_terminal_size()
        column = sz.columns
        row = sz.lines
    except OSError:
        column = 144
        row = 11

    if type(dataset_or_ins).__name__ == "DataSet":
        x.field_names = list(dataset_or_ins.field_arrays.keys())
        c_size = len(x.field_names)
        for ins in dataset_or_ins:
            x.add_row([sub_column(ins[k], column, c_size, k) for k in x.field_names])
            row -= 1
            if row < 0:
                x.add_row(["..." for _ in range(c_size)])
                break
    elif type(dataset_or_ins).__name__ == "Instance":
        x.field_names = list(dataset_or_ins.fields.keys())
        c_size = len(x.field_names)
        x.add_row([sub_column(dataset_or_ins[k], column, c_size, k) for k in x.field_names])

    else:
        raise Exception("only accept  DataSet and Instance")
    x.align = "l"

    return x


def sub_column(string: str, c: int, c_size: int, title: str) -> str:
    r"""
    :param string: 要被截断的字符串
    :param c: 命令行列数
    :param c_size: instance或dataset field数
    :param title: 列名
    :return: 对一个过长的列进行截断的结果
    """
    avg = max(int(c / c_size / 2), len(title))
    string = str(string)
    res = ""
    counter = 0
    for char in string:
        if ord(char) > 255:
            counter += 2
        else:
            counter += 1
        res += char
        if counter > avg:
            res = res + "..."
            break
    return res


def _is_function_contains_autocast(func):
    """
    检查func是否包含autocast，(1)是否使用了autocast的修饰器或, (2)使用使用with autocast()环境

    :param func: 待检查的函数
    """
    import re
    source = inspect.getsource(func)
    lines = source.split('\n')
    for line in lines:
        line = line.strip()
        if re.search(r'@[\w\.]*autocast\(\w*\)', line):
            raise RuntimeError("Please do not use `autocast()` decorator, use `with autocast():` instead. Please refer to"
                               " https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process ")
        if re.search(r'with [\w\.]*autocast\(\w*\):', line):
            return True
    return False


class DummyGradScaler:
    """
    用于Dummy pytorch的GradScaler对象，防止重复写大量的if判断

    """
    def __init__(self, *args, **kwargs):
        pass

    def get_scale(self):
        return 1.0

    def is_enabled(self):
        return False

    def scale(self, outputs):
        return outputs

    def step(self, optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def unscale_(self, optimizer):
        pass

    def load_state_dict(self, state_dict):
        pass

    def state_dict(self):
        return {}


def _build_fp16_env(dummy=False):
    if dummy:
        autocast = contextlib.ExitStack
        GradScaler = DummyGradScaler
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("No cuda")
        if torch.cuda.get_device_capability(0)[0] < 7:
            warnings.warn(
                "NOTE: your device does NOT support faster training with fp16, "
                "please switch to FP32 which is likely to be faster"
            )
        try:
            from torch.cuda.amp import autocast, GradScaler
        except ImportError:
            raise RuntimeError("torch version too low (less than 1.6)")
    return autocast, GradScaler


def _can_use_fp16(device, model, func):
    if parse_version(torch.__version__) < parse_version('1.6'):
        raise RuntimeError("Pytorch supports float16 after version 1.6, please upgrade your pytorch version.")
    model_device = _get_model_device(model)
    if device is None and model_device is not None and model_device.type != 'cuda':
        raise RuntimeError("You have to run in cuda device to use fp16.")
    if isinstance(device, str):
        if device=='cpu':
            raise RuntimeError("You have to run in cuda device to use fp16.")
    if isinstance(device, torch.device) and device.type=='cpu':
        raise RuntimeError("You have to run in cuda device to use fp16.")

    if (_model_contains_inner_module(model) or (isinstance(device, list) and len(device) > 1)):
        # 需要提醒用户
        if not _is_function_contains_autocast(func):
            raise RuntimeError("When use fp16 in Parallel Training, you have to set autocast() in your forward "
                               "function as described in "
                               "https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process")

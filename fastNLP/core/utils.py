"""
utils模块实现了 fastNLP 内部和外部所需的很多工具。其中用户可以使用的是 :func:`cache_results` 修饰器。
"""
__all__ = [
    "cache_results",
    "seq_len_to_mask",
    "Example",
]

import _pickle
import inspect
import os
import warnings
from collections import Counter, namedtuple

import numpy as np
import torch
import torch.nn as nn


_CheckRes = namedtuple('_CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                     'varargs'])

from copy import deepcopy
from .vocabulary import Vocabulary

class ConfusionMatrix:
    def __init__(self, vocab=None):
        if  vocab and not isinstance(vocab,Vocabulary):
            raise TypeError(f"`vocab` in {_get_func_signature(self.__init__)} must be Fastnlp.core.Vocabulary,"
                            f"got {type(vocab)}.")
        self.confusiondict={} #key: vocab的index, value: word ocunt
        # self.predcount={}  #key:pred index, value:count
        self.targetcount={}
        self.vocab=vocab
        

    def add_pred_target(self, pred, target):  #一组结果
        """
        通过这个函数向ConfusionMatrix加入一组预测结果

        :param list pred:
        :param list target:
        :return:
        """
        for p,t in zip(pred,target): #<int, int>
            # self.predcount[p]=self.predconut.get(p,0)+ 1
            self.targetcount[t]=self.targetcount.get(t,0)+1 
            if p in self.confusiondict:
                self.confusiondict[p][t]=self.confusiondict[p].get(t,0) + 1
            else:
                self.confusiondict[p]={}
                self.confusiondict[p][t]= 1
            
        return self.confusiondict

        
    
    def clear(self):
        """
        清除一些值，等待再次新加入
        
        :return: 
        """
        self.confusiondict={}
        self.targetcount={}
        
    
    def __repr__(self):
        row2idx={}
        idx2row={}
        # targetdict=sorted(self.confusiondict.items()) # key:index value:wordcount  5,1,3...->1,3,5...
        # targetdict=dict(targetdict)
        s=len(self.targetcount)
        namedict={}  # key :idx value:word/idx

        if self.vocab:
            namedict=self.vocab.idx2word
        else:
            namedict=dict([(k,str(k)) for k in self.targetcount.keys()])
            # 这里打印东西

        for key,num in zip(self.targetcount.keys(),range(s)):
            idx2row[key]=num  #建立一个临时字典，key:vocab的index, value: 行列index  1,3,5...->0,1,2,...
            row2idx[num]=key  #建立一个临时字典，value:vocab的index, key: 行列index  0,1,2...->1,3,5,...

        head=["\ntarget"]+[str(namedict[k]) for k in row2idx.keys()]+["all"]
        output="\t".join(head) + "\n" + "pred" + "\n"
        for i in row2idx.keys(): #第i行
            p=row2idx[i]
            h=namedict[p]
            l=[0 for _ in range(s)]
            if self.confusiondict.get(p,None):
                # print(type(self.confusiondict[p]))
                for t,c in self.confusiondict[p].items():
                    l[idx2row[t]] = c #完成一行
            #增加表头
            l=[h]+[str(n) for n in l]+[str(sum(l))]
            output+="\t".join(l) +"\n"
        tail=[self.targetcount.get(k,0) for k in row2idx.keys()]
        tail=["all"]+[str(n) for n in tail]+[str(sum(tail))]
        output+="\t".join(tail)
        return output

class ConfusionMatrixMetric():
    def __init__(self, vocab=None, pred='pred', target='target', seq_len='seq_len'):
        # super().__init__()
        # self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.confusion_matrix = ConfusionMatrix(vocab=vocab)

    def evaluate(self, pred, target, seq_len=None):
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_len is not None and not isinstance(seq_len, torch.Tensor):
            raise TypeError(f"`seq_lens` in {_get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_len)}.")

        if pred.dim() == target.dim():
            pass
        elif pred.dim() == target.dim() + 1:
            pred = pred.argmax(dim=-1)
            if seq_len is None and target.dim() > 1:
                warnings.warn("You are not passing `seq_len` to exclude pad.")
        else:
            raise RuntimeError(f"In {_get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        target = target.to(pred)
        if seq_len is not None and  target.dim() > 1:
            for p, t, l in zip(pred.tolist(), target.tolist(), seq_len.tolist()):  
                self.confusion_matrix.add_pred_target(p[::l], t[::l])
        else:
            self.confusion_matrix.add_pred_target(pred.tolist(), target.tolist())



    def get_metric(self,reset=True):
        confusion = {'confusion_matrix': deepcopy(self.confusion_matrix)}
        if reset:
            self.confusion_matrix.clear()
        return confusion


class Example(dict):
    """a dict can treat keys as attributes"""
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
    """
    检查filepath是否可以作为合理的cache文件. 如果可以的话，会自动创造路径
    :param filepath: str.
    :return: None, if not, this function will raise error
    """
    _cache_filepath = os.path.abspath(filepath)
    if os.path.isdir(_cache_filepath):
        raise RuntimeError("The cache_file_path must be a file, not a directory.")
    cache_dir = os.path.dirname(_cache_filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


#  TODO 可以保存下缓存时的参数，如果load的时候发现参数不一致，发出警告。
def cache_results(_cache_fp, _refresh=False, _verbose=1):
    """
    别名：:class:`fastNLP.cache_results` :class:`fastNLP.core.uitls.cache_results`

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
                        print("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False
            
            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    _prepare_cache_filepath(cache_filepath)
                    with open(cache_filepath, 'wb') as f:
                        _pickle.dump(results, f)
                    print("Save cache to {}.".format(cache_filepath))
            
            return results
        
        return wrapper
    
    return wrapper_


# def save_pickle(obj, pickle_path, file_name):
#     """Save an object into a pickle file.
#
#     :param obj: an object
#     :param pickle_path: str, the directory where the pickle file is to be saved
#     :param file_name: str, the name of the pickle file. In general, it should be ended by "pkl".
#     """
#     if not os.path.exists(pickle_path):
#         os.mkdir(pickle_path)
#         print("make dir {} before saving pickle file".format(pickle_path))
#     with open(os.path.join(pickle_path, file_name), "wb") as f:
#         _pickle.dump(obj, f)
#     print("{} saved in {}".format(file_name, pickle_path))
#
#
# def load_pickle(pickle_path, file_name):
#     """Load an object from a given pickle file.
#
#     :param pickle_path: str, the directory where the pickle file is.
#     :param file_name: str, the name of the pickle file.
#     :return obj: an object stored in the pickle
#     """
#     with open(os.path.join(pickle_path, file_name), "rb") as f:
#         obj = _pickle.load(f)
#     print("{} loaded from {}".format(file_name, pickle_path))
#     return obj
#
#
# def pickle_exist(pickle_path, pickle_name):
#     """Check if a given pickle file exists in the directory.
#
#     :param pickle_path: the directory of target pickle file
#     :param pickle_name: the filename of target pickle file
#     :return: True if file exists else False
#     """
#     if not os.path.exists(pickle_path):
#         os.makedirs(pickle_path)
#     file_name = os.path.join(pickle_path, pickle_name)
#     if os.path.exists(file_name):
#         return True
#     else:
#         return False

def _move_model_to_device(model, device):
    """
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
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        raise RuntimeError("model of `torch.nn.parallel.DistributedDataParallel` is not supported right now.")
    
    if device is None:
        if isinstance(model, torch.nn.DataParallel):
            model.cuda()
        return model
    else:
        if not torch.cuda.is_available() and (
                device != 'cpu' or (isinstance(device, torch.device) and device.type != 'cpu')):
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
    """
    传入一个nn.Module的模型，获取它所在的device

    :param model: nn.Module
    :return: torch.device,None 如果返回值为None，说明这个模型没有任何参数。
    """
    assert isinstance(model, nn.Module)
    
    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _build_args(func, **kwargs):
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
            # TODO: add UNUSED warning.
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
    """

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
    """

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
    """

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param non_blocking: bool, 是否异步将数据转移到cpu, 需要tensor使用pin_memory()
    :param args:
    :return:
    """
    if not torch.cuda.is_available():
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
    """

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
                    _tmp += f' Or provide `{_miss}` in DataSet or output of {prev_func_signature}.'
                else:
                    _tmp = f'Provide `{_miss}` in DataSet or output of {prev_func_signature}.'
                suggestions.append(_tmp)
        # for _miss in unmapped_missing:
        #     if _miss in dataset:
        #         suggestions.append(f"Set `{_miss}` as target.")
        #     else:
        #         _tmp = ''
        #         if check_res.unused:
        #             _tmp = f"Specify your assignment for `{input_func_map.get(_miss, _miss)}` when initialize {module_name}."
        #         if _tmp:
        #             _tmp += f' Or provide `{_miss}` in DataSet or output of {prev_func_signature}.'
        #         else:
        #             _tmp = f'Provide `{_miss}` in output of {prev_func_signature} or DataSet.'
        #         suggestions.append(_tmp)
    
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
            suggestions.append(f"You might need to set {_miss_in_dataset} as input. ")
        if _miss_out_dataset:
            _tmp = f"You need to provide {_miss_out_dataset} in DataSet and set it as input. "
            # if check_res.unused:
            #     _tmp += f"Or you might find it in `unused field:`, you can use DataSet.rename_field() to " \
            #             f"rename the field in `unused field:`."
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
        else:
            sugg_str += suggestions[0]
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def seq_len_to_mask(seq_len):
    """

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    Example::
    
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :return: np.ndarray or torch.Tensor, shape将是(B, max_length)。 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
    
    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    
    return mask


class _pseudo_tqdm:
    """
    当无法引入tqdm，或者Trainer中设置use_tqdm为false的时候，用该方法打印数据
    """
    
    def __init__(self, **kwargs):
        pass
    
    def write(self, info):
        print(info)
    
    def set_postfix_str(self, info):
        print(info)
    
    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass
        
        return pass_func
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

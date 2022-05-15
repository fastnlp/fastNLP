import functools
import inspect
from inspect import Parameter
import dataclasses
import warnings
from dataclasses import is_dataclass
from copy import deepcopy
from collections import defaultdict, OrderedDict
from typing import Callable, List, Any, Dict, AnyStr, Union, Mapping, Sequence
from typing import Tuple, Optional
from time import sleep

import os
from contextlib import contextmanager
from functools import wraps
from prettytable import PrettyTable
import numpy as np
from pathlib import Path

from fastNLP.core.log import logger


__all__ = [
    'get_fn_arg_names',
    'auto_param_call',
    'check_user_specific_params',
    'dataclass_to_dict',
    'match_and_substitute_params',
    'apply_to_collection',
    'nullcontext',
    'pretty_table_printer',
    'Option',
    'deprecated',
    'seq_len_to_mask',
    "flat_nest_dict"
]


def get_fn_arg_names(fn: Callable) -> List[str]:
    r"""
    该函数可以返回一个函数所有参数的名字::

        >>> def function(a, b=1):
        ...     return a
        ...
        >>> get_fn_arg_names(function)
        ['a', 'b']

    :param fn: 需要查询的函数；
    :return: 包含函数 ``fn`` 参数名的列表；
    """
    return list(inspect.signature(fn).parameters)


def auto_param_call(fn: Callable, *args, signature_fn: Optional[Callable] = None,
                    mapping: Optional[Dict[AnyStr, AnyStr]] = None) -> Any:
    r"""
    该函数会根据输入函数的形参名从 ``*args`` （均为 **dict** 类型）中找到匹配的值进行调用，如果传入的数据与 ``fn`` 的形参不匹配，可以通过
    ``mapping`` 参数进行转换。``mapping`` 参数中的一对 ``(key, value)`` 表示在 ``*args`` 中找到 ``key`` 对应的值，并将这个值传递给形参中名为
    ``value`` 的参数。

    1. 该函数用来提供给用户根据字符串匹配从而实现自动调用；
    2. 注意 ``mapping`` 默认为 ``None``，如果您希望指定输入和运行函数的参数的对应方式，那么您应当让 ``mapping`` 为一个字典传入进来；
       如果 ``mapping`` 不为 ``None``，那么我们一定会先使用 ``mapping`` 将输入的字典的 ``keys`` 修改过来，因此请务必亲自检查 ``mapping`` 的正确性；
    3. 如果输入的函数的参数有默认值，那么如果之后的输入中没有该参数对应的值，我们就会使用该参数对应的默认值，否则也会使用之后的输入的值；
    4. 如果输入的函数是一个 ``partial`` 函数，情况同第三点，即和默认参数的情况相同；

    Examples::

        >>> # 1
        >>> loss_fn = CrossEntropyLoss()  # 如果其需要的参数为 def CrossEntropyLoss(y, pred)；
        >>> batch = {"x": 20, "y": 1}
        >>> output = {"pred": 0}
        >>> acc = auto_param_call(loss_fn, batch, output)

        >>> # 2
        >>> def test_fn(x, y, a, b=10):
        >>>     return x + y + a + b
        >>> print(auto_param_call(test_fn, {"x": 10}, {"y": 20, "a": 30}))  # res: 70
        >>> print(auto_param_call(partial(test_fn, a=100), {"x": 10}, {"y": 20}))  # res: 140
        >>> print(auto_param_call(partial(test_fn, a=100), {"x": 10}, {"y": 20, "a": 200}))  # res: 240

    :param fn: 用来进行实际计算的函数，其参数可以包含有默认值；
    :param args: 一系列的位置参数，应当为一系列的字典，我们需要从这些输入中提取 ``fn`` 计算所需要的实际参数；
    :param signature_fn: 函数，用来替换 ``fn`` 的函数签名，如果该参数不为 ``None``，那么我们首先会从该函数中提取函数签名，
        然后通过该函数签名提取参数值后，再传给 ``fn`` 进行实际的运算；
    :param mapping: 一个字典，用来更改其前面的字典的键值；

    :return: 返回 ``fn`` 运行的结果；
    """

    if signature_fn is not None:
        if not callable(signature_fn):
            raise ValueError(f"Parameter `signature_fn` should be `Callable`.")
        _need_params = OrderedDict(inspect.signature(signature_fn).parameters)
    else:
        _need_params = OrderedDict(inspect.signature(fn).parameters)
    _kwargs = None
    for _name, _param in _need_params.items():
        if _param.kind == Parameter.VAR_POSITIONAL:
            fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
            raise ValueError(f"It is not allowed to have parameter `*args` in your function:{fn_msg}.")
        if _param.kind == Parameter.VAR_KEYWORD:
            _kwargs = (_name, _param)

    if _kwargs is not None:
        _need_params.pop(_kwargs[0])

    _default_params = {}
    for _name, _param in _need_params.items():
        if _param.default != Parameter.empty:
            _default_params[_name] = _param.default

    if mapping is not None:
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        assert isinstance(mapping, Dict), f"Exception happens when calling {fn_msg}. " \
                                          f"Parameter `mapping` should be of 'Dict' type, instead of {type(mapping)}."

    _has_params = {}
    duplicate_names = []
    for arg in args:
        if not isinstance(arg, Dict):
            fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
            raise TypeError(f"Exception happens when calling {fn_msg}. "
                            f"The input part of function `auto_param_call` must be `Dict` type, instead of {type(arg)}.")
        for _name, _value in arg.items():
            if mapping is not None and _name in mapping:
                _name = mapping[_name]

            if _name not in _has_params:
                if _kwargs is not None or _name in _need_params:
                    _has_params[_name] = _value
            # 同一参数对象在两个输入的资源中都出现，造成混淆；
            elif _name in _need_params and not (_has_params[_name] is _value):
                duplicate_names.append(_name)
    if duplicate_names:
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        raise ValueError(f"The following key present in several inputs:{duplicate_names} when calling {fn_msg}.")

    # 将具有默认值但是没有被输入修改过的参数值传进去；
    for _name, _value in _default_params.items():
        if _name not in _has_params:
            _has_params[_name] = _value

    if len(_has_params) < len(_need_params):
        miss_params = list(set(_need_params.keys()) - set(_has_params.keys()))
        fn_msg = _get_fun_msg(fn if signature_fn is None else signature_fn)
        _provided_keys = _get_keys(args)
        raise ValueError(f"The parameters:`{miss_params}` needed by function:{fn_msg} "
                         f"are not found in the input keys({_provided_keys}).")

    return fn(**_has_params)


def _get_keys(args:List[Dict]) -> List[List[str]]:
    """
    返回每个 dict 的 keys

    :param args:
    :return:
    """
    _provided_keys = []
    for arg in args:
        _provided_keys.append(list(arg.keys()))
    return _provided_keys


def _get_fun_msg(fn, with_fp=True)->str:
    """
    获取函数的基本信息，帮助报错::

        >>>> print(_get_fun_msg(_get_fun_msg))
        `_get_fun_msg(fn) -> str`(In file:/Users/hnyan/Desktop/projects/fastNLP/fastNLP/fastNLP/core/utils/utils.py)

    :param callable fn:
    :param with_fp: 是否包含函数所在的文件信息；
    :return:
    """
    if isinstance(fn, functools.partial):
        return _get_fun_msg(fn.func)
    try:
        fn_name = fn.__qualname__ + str(inspect.signature(fn))
    except:
        fn_name = str(fn)
    if with_fp:
        try:
            fp = '(In file:' + os.path.abspath(inspect.getfile(fn)) + ')'
        except:
            fp = ''
    else:
        fp = ''
    msg = f'`{fn_name}`' + fp
    return msg


def _check_valid_parameters_number(fn, expected_params:List[str], fn_name=None):
    """
    检查一个函数是否需要 expected_params 参数(检测数量是否匹配)。除掉 self （如果是method），给定默认值的参数等。
    如果匹配不上，就会进行报错。

    :param fn: 需要检测的函数，可以是 method 或者 function 。
    :param expected_params: 期待应该支持的参数。
    :param fn_name: fn 的名字，当传入的 fn 不是 callable 的时候方便报错。
    :return:
    """
    if fn_name is not None:
        assert callable(fn), f"`{fn_name}` should be callable, instead of `{type(fn)}`."

    try:
        args = []
        kwargs = {}
        name = ''
        if isinstance(fn, functools.partial) and not hasattr(fn, '__name__'):
            name = 'partial:'
            f = fn.func
            while isinstance(f, functools.partial):
                name += 'partial:'
                f = f.func
            fn.__name__ = name + f.__name__
        inspect.getcallargs(fn, *args, *expected_params, **kwargs)
        if name:  # 如果一开始没有name的，需要给人家删除掉
            delattr(fn, '__name__')

    except TypeError as e:
        logger.error(f"The function:{_get_fun_msg(fn)} will be provided with parameters:{expected_params}. "
                     f"The following exception will happen.")
        raise e


def check_user_specific_params(user_params: Dict, fn: Callable):
    """
    该函数使用用户的输入来对指定函数的参数进行赋值，主要用于一些用户无法直接调用函数的情况；
    主要作用在于帮助检查用户对使用函数 ``fn`` 的参数输入是否有误；

    :param user_params: 用户指定的参数的值，应当是一个字典，其中 ``key`` 表示每一个参数的名字，
        ``value`` 为每一个参数的值；
    :param fn: 将要被调用的函数；
    :return: 返回一个字典，其中为在之后调用函数 ``fn`` 时真正会被传进去的参数的值；
    """

    fn_arg_names = get_fn_arg_names(fn)
    for arg_name, arg_value in user_params.items():
        if arg_name not in fn_arg_names:
            logger.rank_zero_warning(f"Notice your specific parameter `{arg_name}` is not used by function `{fn.__name__}`.")
    return user_params


def dataclass_to_dict(data: "dataclasses.dataclass") -> Dict:
    """
    将传入的 ``dataclass`` 实例转换为字典。
    """
    if not is_dataclass(data):
        raise TypeError(f"Parameter `data` can only be `dataclass` type instead of {type(data)}.")
    _dict = dict()
    for _key in data.__dataclass_fields__:
        _dict[_key] = getattr(data, _key)
    return _dict


def match_and_substitute_params(mapping: Optional[Union[Callable, Dict]] = None, data: Optional[Any] = None) -> Any:
    r"""
    用来实现将输入的 **batch** 或者输出的 **outputs** 通过 ``mapping`` 将键值进行更换的功能；
    该函数应用于 ``input_mapping`` 和 ``output_mapping``；

    * 对于 ``input_mapping``，该函数会在 :class:`~fastNLP.core.controllers.TrainBatchLoop` 中取完数据后立刻被调用；
    * 对于 ``output_mapping``，该函数会在 :class:`~fastNLP.core.Trainer` 的 :meth:`~fastNLP.core.Trainer.train_step`
      以及 :class:`~fastNLP.core.Evaluator` 的 :meth:`~fastNLP.core.Evaluator.train_step` 中得到结果后立刻被调用；

    转换的逻辑按优先级依次为：

    1. 如果 ``mapping`` 是一个函数，那么会直接返回 **mapping(data)**；
    2. 如果 ``mapping`` 是一个 **Dict**，那么 ``data`` 的类型只能为以下三种： ``[Dict, dataclass, Sequence]``；
        
        * 如果 ``data`` 是 **Dict**，那么该函数会将 ``data`` 的 ``key`` 替换为 **mapping[key]**；
        * 如果 ``data`` 是 **dataclass**，那么该函数会先使用 :func:`dataclasses.asdict` 函数将其转换为 **Dict**，然后进行转换；
        * 如果 ``data`` 是 **Sequence**，那么该函数会先将其转换成一个对应的字典::
        
            {
                "_0": list[0],
                "_1": list[1],
                ...
            }

          然后使用 ``mapping`` 对这个字典进行转换，如果没有匹配上 ``mapping`` 中的 ``key`` 则保持 ``'_number'`` 这个形式。

    :param mapping: 用于转换的字典或者函数；当 ``mapping`` 是函数时，返回值必须为字典类型；
    :param data: 需要被转换的对象；
    :return: 返回转换后的结果；
    """
    if mapping is None:
        return data
    if callable(mapping):
        # 注意我们在 `Trainer.extract_loss_from_outputs` 函数里会检查 outputs 的输出，outputs 的类型目前只支持 `Dict` 和 `dataclass`；
        return mapping(data)

    if not isinstance(mapping, Dict):
        raise ValueError(
            f"Parameter `mapping` should be of type `Dict` or `Callable`, not `{type(mapping)}`. This is caused"
            f"by your `input_mapping` or `output_mapping` parameter in your `Trainer` or `Evaluator`.")
    if not isinstance(data, Dict) and not is_dataclass(data) and not isinstance(data, Sequence):
        raise ValueError("Parameter `data` should be type `Dict` or `dataclass` when the other parameter `mapping` is "
                         "type `Dict`.")

    # 如果 `data` 是一个 dataclass，那么先将其转换为一个 `Dict`；
    if is_dataclass(data):
        data = dataclass_to_dict(data)
    # 如果 `data` 是一个 List，那么我们同样先将其转换为一个 `Dict`，为 {"_0": list[0], "_1": list[1], ...}；
    elif isinstance(data, Sequence):
        data = {"_" + str(i): data[i] for i in range(len(data))}

    _new_data = {}
    for _name, _value in data.items():
        if _name in mapping:
            _new_data[mapping[_name]] = _value
        else:
            _new_data[_name] = _value
    return _new_data


def _is_namedtuple(obj: object) -> bool:
    # https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/parallel/scatter_gather.py#L4-L8
    return isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")


def _is_dataclass_instance(obj: object) -> bool:
    # https://docs.python.org/3/library/dataclasses.html#module-level-decorators-classes-and-functions
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def apply_to_collection(
        data: Any,
        dtype: Union[type, Any, Tuple[Union[type, Any]]],
        function: Callable,
        *args: Any,
        wrong_dtype: Optional[Union[type, Tuple[type]]] = None,
        include_none: bool = True,
        **kwargs: Any,
) -> Any:
    """
    递归地对 ``data`` 中的元素执行函数 ``function``，且仅在满足元素为 ``dtype`` 时执行。

    该函数参考了 `pytorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ 的实现

    :param data: 需要进行处理的数据集合或数据；
    :param dtype: 数据的类型，函数 ``function`` 只会被应用于 ``data`` 中类型为 ``dtype`` 的数据；
    :param function: 对数据进行处理的函数；
    :param args: ``function`` 所需要的其它参数；
    :param wrong_dtype: ``function`` 一定不会生效的数据类型。
        如果数据既是 ``wrong_dtype`` 类型又是 ``dtype`` 类型那么也不会生效；
    :param include_none: 是否包含执行结果为 ``None`` 的数据，默认为 ``True``；
    :param kwargs: ``function`` 所需要的其它参数；
    :return: 经过 ``function`` 处理后的数据集合；
    """
    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    elem_type = type(data)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        out = []
        for k, v in data.items():
            v = apply_to_collection(
                v, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append((k, v))
        if isinstance(data, defaultdict):
            return elem_type(data.default_factory, OrderedDict(out))
        return elem_type(OrderedDict(out))

    is_namedtuple = _is_namedtuple(data)
    is_sequence = isinstance(data, Sequence) and not isinstance(data, str)
    if is_namedtuple or is_sequence:
        out = []
        for d in data:
            v = apply_to_collection(
                d, dtype, function, *args, wrong_dtype=wrong_dtype, include_none=include_none, **kwargs
            )
            if include_none or v is not None:
                out.append(v)
        return elem_type(*out) if is_namedtuple else elem_type(out)

    if _is_dataclass_instance(data):
        # make a deepcopy of the data,
        # but do not deepcopy mapped fields since the computation would
        # be wasted on values that likely get immediately overwritten
        fields = {}
        memo = {}
        for field in dataclasses.fields(data):
            field_value = getattr(data, field.name)
            fields[field.name] = (field_value, field.init)
            memo[id(field_value)] = field_value
        result = deepcopy(data, memo=memo)
        # apply function to each field
        for field_name, (field_value, field_init) in fields.items():
            if field_init:
                v = apply_to_collection(
                    field_value,
                    dtype,
                    function,
                    *args,
                    wrong_dtype=wrong_dtype,
                    include_none=include_none,
                    **kwargs,
                )
            if not field_init or (not include_none and v is None):  # retain old value
                v = getattr(data, field_name)
            setattr(result, field_name, v)
        return result

    # data is neither of dtype, nor a collection
    return data


@contextmanager
def nullcontext():
    r"""
    实现一个什么都不做的上下文环境。
    """
    yield


def sub_column(string: str, c: int, c_size: int, title: str) -> str:
    r"""
    对传入的字符串进行截断，方便在命令行中显示。

    :param string: 要被截断的字符串；
    :param c: 命令行列数；
    :param c_size: :class:`~fastNLP.core.Instance` 或 :class:`~fastNLP.core.DataSet` 的 ``field`` 数目；
    :param title: 列名；
    :return: 对一个过长的列进行截断的结果；
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


def _is_iterable(value):
    # 检查是否是iterable的, duck typing
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


def pretty_table_printer(dataset_or_ins) -> PrettyTable:
    r"""
    用于在 **fastNLP** 中展示数据的函数::

        >>> ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2], field_3=["a", "b", "c"])
        +-----------+-----------+-----------------+
        |  field_1  |  field_2  |     field_3     |
        +-----------+-----------+-----------------+
        | [1, 1, 1] | [2, 2, 2] | ['a', 'b', 'c'] |
        +-----------+-----------+-----------------+

    :param dataset_or_ins: 要展示的 :class:`~fastNLP.core.DataSet` 或者 :class:`~fastNLP.core.Instance` 实例；
    :return: 根据命令行大小进行自动截断的数据表格；
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


class Option(dict):
    r"""将键转化为属性的字典类型"""

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


_emitted_deprecation_warnings = set()


def deprecated(help_message: Optional[str] = None):
    """
    标记当前功能已经过时的装饰器。

    :param help_message: 一段指引信息，告知用户如何将代码切换为当前版本提倡的用法；
    """

    def decorator(deprecated_function: Callable):
        global _emitted_deprecation_warnings
        warning_msg = (
            (
                f"{deprecated_function.__name__} is deprecated and will be removed "
                "in the next major version of datasets."
            )
            + f" {help_message}"
            if help_message
            else ""
        )

        @wraps(deprecated_function)
        def wrapper(*args, **kwargs):
            func_hash = hash(deprecated_function)
            if func_hash not in _emitted_deprecation_warnings:
                warnings.warn(warning_msg, category=FutureWarning, stacklevel=2)
                _emitted_deprecation_warnings.add(func_hash)
            return deprecated_function(*args, **kwargs)

        wrapper._decorator_name_ = "deprecated"
        return wrapper

    return decorator


def seq_len_to_mask(seq_len, max_len: Optional[int]):
    r"""

    将一个表示 ``sequence length`` 的一维数组转换为二维的 ``mask`` ，不包含的位置为 **0**。

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

    :param seq_len: 大小为 ``(B,)`` 的长度序列；
    :param int max_len: 将长度补齐或截断到 ``max_len``。默认情况（为 ``None``）使用的是 ``seq_len`` 中最长的长度；
        但在 :class:`torch.nn.DataParallel` 等分布式的场景下可能不同卡的 ``seq_len`` 会有区别，所以需要传入
        ``max_len`` 使得 ``mask`` 的补齐或截断到该长度。
    :return: 大小为 ``(B, max_len)`` 的 ``mask``， 元素类型为 ``bool`` 或 ``uint8``
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    else:
        raise TypeError("Only support 1-d numpy.ndarray.")

    return mask


def wait_filepath(path, exist=True):
    """
    等待当 path 的存在状态为 {exist} 时返回

    :param path: 待检测的 path
    :param exist: 为 True 时表明检测这个 path 存在就返回; 为 False 表明检测到这个 path 不存在 返回。
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    assert isinstance(path, Path)
    count = 0
    while True:
        sleep(0.01)
        if path.exists() == exist:
            break
        count += 1
        if count % 1000 == 0:
            msg = 'create' if exist else 'delete'
            logger.warning(f"Waiting path:{path} to {msg} for {count*0.01} seconds...")


def get_class_that_defined_method(method):
    """
    给定一个method，返回这个 method 的 class 的对象

    :param method:
    :return:
    """
    if isinstance(method, functools.partial):
        return get_class_that_defined_method(method.func)
    if inspect.ismethod(method) or (inspect.isbuiltin(method) and getattr(method, '__self__', None) is not None and getattr(method.__self__, '__class__', None)):
        for cls in inspect.getmro(method.__self__.__class__):
            if method.__name__ in cls.__dict__:
                return cls
        method = getattr(method, '__func__', method)  # fallback to __qualname__ parsing
    if inspect.isfunction(method):
        cls = getattr(inspect.getmodule(method),
                      method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(method, '__objclass__', None)  # handle special descriptor objects


def is_notebook():
    """
    检查当前运行环境是否为 jupyter

    :return:
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
    except:
        return False
    else:  # pragma: no cover
        return True


def flat_nest_dict(d:Dict, separator:str='#', compress_none_key:bool=True, top_down:bool=False) -> Dict:
    """
    讲一个 nested 的 dict 转成 flat 的 dict，例如
    ex::
        d = {'test': {'f1': {'f': 0.2, 'rec': 0.1}}} -> {'f#f1#test':0.2, 'rec#f1#test':0.1}

    :param d: 需要展平的 dict 对象。
    :param separator: 不同层级之间的 key 之间的连接符号。
    :param compress_none_key: 如果有 key 为 None ，则忽略这一层连接。
    :param top_down: 新的 key 的是否按照从最底层往最底层的顺序连接。
    :return:
    """
    assert isinstance(d, Dict)
    assert isinstance(separator, str)
    flat_d = {}
    for key, value in d.items():
        if key is None:
            key = ()
        else:
            key = (key, )
        if isinstance(value, Mapping):
            flat_d.update(_flat_nest_dict(value, parent_key=key, compress_none_key=compress_none_key))
        else:
            flat_d[key] = value

    str_flat_d = {}
    for key, value in flat_d.items():
        if top_down:
            key = map(str, key)
        else:
            key = map(str, key[::-1])
        key = separator.join(key)
        str_flat_d[key] = value
    return str_flat_d


def _flat_nest_dict(d:Mapping, parent_key:Tuple, compress_none_key:bool):
    flat_d = {}
    for k, v in d.items():
        _key = parent_key
        if k is not None:
            _key = _key + (k,)
        if isinstance(v, Mapping):
            _d = _flat_nest_dict(v, parent_key=_key, compress_none_key=compress_none_key)
            flat_d.update(_d)
        else:
            flat_d[_key] = v

    return flat_d

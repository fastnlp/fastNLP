import inspect
from inspect import Parameter
import dataclasses
import warnings
from dataclasses import is_dataclass
from copy import deepcopy
from collections import defaultdict, OrderedDict
from typing import Callable, List, Any, Dict, AnyStr, Union, Mapping, Sequence, Optional
from typing import Tuple, Optional
from time import sleep

try:
    from typing import Literal, Final
except ImportError:
    from typing_extensions import Literal, Final
import os
from contextlib import contextmanager
from functools import wraps
from prettytable import PrettyTable
import numpy as np
from pathlib import Path

from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_GLOBAL_RANK



__all__ = [
    'get_fn_arg_names',
    'check_fn_not_empty_params',
    'auto_param_call',
    'check_user_specific_params',
    'dataclass_to_dict',
    'match_and_substitute_params',
    'apply_to_collection',
    'nullcontext',
    'pretty_table_printer',
    'Option',
    'indice_collate_wrapper',
    'deprecated',
    'seq_len_to_mask',
    'synchronize_safe_rm',
    'synchronize_mkdir'
]


def get_fn_arg_names(fn: Callable) -> List[str]:
    r"""
    返回一个函数的所有参数的名字；

    :param fn: 需要查询的函数；

    :return: 一个列表，其中的元素则是查询函数的参数的字符串名字；
    """
    return list(inspect.signature(fn).parameters)


def check_fn_not_empty_params(fn: Optional[Callable] = None, param_num: Optional[int] = None) -> bool:
    r"""
    检查传入的batch_step_fn是否是合法的：(1) 是否是 callable 的; (2) 没有默认值的参数是否只有指定个数；
    用户也可以传进一个 partial 的函数进来，只要其保证留有 `trainer` 和 `batch` 的参数位置即可；

    :param fn: 传入的用以代替 Loop 中 'step' 函数的函数；
    :param param_num: 检测的函数的应当的没有默认值的参数的个数；

    :return: bool，表示传入的 `batch_step_fn` 是否正确；
    """

    if fn is None:
        return True
    if not callable(fn):
        return False
    else:
        params = inspect.signature(fn).parameters
        not_default_params = {}
        for _name, _param in params.items():
            if _param.default == Parameter.empty:
                not_default_params[_name] = _param
        return len(not_default_params) == param_num


def auto_param_call(fn: Callable, *args, signature_fn: Optional[Callable] = None,
                    mapping: Optional[Dict[AnyStr, AnyStr]] = None) -> Any:
    r"""
    1.该函数用来提供给用户根据字符串匹配从而实现自动计算；
    2.注意 mapping 默认为 None，如果你希望指定输入和运行函数的参数的对应方式，那么你应当让 mapping 为一个这样的字典传入进来；
     如果 mapping 不为 None，那么我们一定会先使用 mapping 将输入的字典的 keys 修改过来，因此请务必亲自检查 mapping 的正确性；
    3.如果输入的函数的参数有默认值，那么如果之后的输入中没有该参数对应的值，我们就会使用该参数对应的默认值，否则也会使用之后的输入的值；
    4.如果输入的函数是一个 `partial` 函数，情况同 '3.'，即和默认参数的情况相同；

    :param fn: 用来进行实际计算的函数，其参数可以包含有默认值；
    :param args: 一系列的位置参数，应当为一系列的字典，我们需要从这些输入中提取 `fn` 计算所需要的实际参数；
    :param signature_fn: 函数，用来替换 `fn` 的函数签名，如果该参数不为 None，那么我们首先会从该函数中提取函数签名，然后通过该函数签名提取
     参数值后，再传给 `fn` 进行实际的运算；
    :param mapping: 一个字典，用来更改其前面的字典的键值；

    :return: 返回 `fn` 运行的结果；

    Examples:
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
            raise ValueError(f"It is not allowed to have parameter `*args` in your function:{fn.__name__}.")
        if _param.kind == Parameter.VAR_KEYWORD:
            _kwargs = (_name, _param)

    if _kwargs is not None:
        _need_params.pop(_kwargs[0])

    _default_params = {}
    for _name, _param in _need_params.items():
        if _param.default != Parameter.empty:
            _default_params[_name] = _param.default

    if mapping is not None:
        assert isinstance(mapping, Dict), f"Parameter `mapping` should be of 'Dict' type, instead of {type(mapping)}."

    _has_params = {}
    duplicate_names = []
    for arg in args:
        assert isinstance(arg, Dict), "The input part of function `auto_param_call` can only be `Dict` type."
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
        raise ValueError(f"The following key present in several inputs:{duplicate_names}")

    # 将具有默认值但是没有被输入修改过的参数值传进去；
    for _name, _value in _default_params.items():
        if _name not in _has_params:
            _has_params[_name] = _value

    if len(_has_params)<len(_need_params):
        miss_params = list(set(_need_params.keys()) - set(_has_params.keys()))
        raise ValueError(f"The parameters:`{miss_params}` needed by function:{fn.__name__} are not found in the input.")

    return fn(**_has_params)


def check_user_specific_params(user_params: Dict, fn: Callable):
    """
    该函数使用用户的输入来对指定函数的参数进行赋值；
    主要用于一些用户无法直接调用函数的情况；
    该函数主要的作用在于帮助检查用户对使用函数 fn 的参数输入是否有误；

    :param user_params: 用户指定的参数的值，应当是一个字典，其中 key 表示每一个参数的名字，value 为每一个参数应当的值；
    :param fn: 会被调用的函数；
    :return: 返回一个字典，其中为在之后调用函数 fn 时真正会被传进去的参数的值；
    """

    fn_arg_names = get_fn_arg_names(fn)
    for arg_name, arg_value in user_params.items():
        if arg_name not in fn_arg_names:
            logger.warning(f"Notice your specific parameter `{arg_name}` is not used by function `{fn.__name__}`.")
    return user_params


def dataclass_to_dict(data: "dataclass") -> Dict:
    if not is_dataclass(data):
        raise TypeError(f"Parameter `data` can only be `dataclass` type instead of {type(data)}.")
    _dict = dict()
    for _key in data.__dataclass_fields__:
        _dict[_key] = getattr(data, _key)
    return _dict


def match_and_substitute_params(mapping: Optional[Union[Callable, Dict]] = None, data: Optional[Any] = None) -> Any:
    r"""
    用来实现将输入：batch，或者输出：outputs，通过 `mapping` 将键值进行更换的功能；
    该函数应用于 `input_mapping` 和 `output_mapping`；
    对于 `input_mapping`，该函数会在 `TrainBatchLoop` 中取完数据后立刻被调用；
    对于 `output_mapping`，该函数会在 `Trainer.train_step` 以及 `Evaluator.train_step` 中得到结果后立刻被调用；

    转换的逻辑按优先级依次为：
     1. 如果 `mapping` 是一个函数，那么会直接返回 `mapping(data)`；
     2. 如果 `mapping` 是一个 `Dict`，那么 `data` 的类型只能为以下三种： [`Dict`, `dataclass`, `Sequence`]；
      如果 `data` 是 `Dict`，那么该函数会将 `data` 的 key 替换为 mapping[key]；
      如果 `data` 是 `dataclass`，那么该函数会先使用 `dataclasses.asdict` 函数将其转换为 `Dict`，然后进行转换；
      如果 `data` 是 `Sequence`，那么该函数会先将其转换成一个对应的 `Dict`：{"_0": list[0], "_1": list[1], ...}，然后使用
        mapping对这个 `Dict` 进行转换，如果没有匹配上mapping中的key则保持"_number"这个形式。

    :param mapping: 用于转换的字典或者函数；mapping是函数时，返回值必须为字典类型。
    :param data: 需要被转换的对象；
    :return: 返回转换好的结果；
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
    """将函数 function 递归地在 data 中的元素执行，但是仅在满足元素为 dtype 时执行。

    this function credit to: https://github.com/PyTorchLightning/pytorch-lightning
    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections
            is of the ``wrong_dtype`` even if it is of type ``dtype``
        include_none: Whether to include an element if the output of ``function`` is ``None``.
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        The resulting collection
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
    用来实现一个什么 dummy 的 context 上下文环境；
    """
    yield


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


def _is_iterable(value):
    # 检查是否是iterable的, duck typing
    try:
        iter(value)
        return True
    except BaseException as e:
        return False


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


def indice_collate_wrapper(func):
    """
    其功能是封装一层collate_fn,将dataset取到的tuple数据分离开，将idx打包为indices。

    :param func: 需要修饰的函数
    :return:
    """

    def wrapper(tuple_data):
        indice, ins_list = [], []
        for idx, ins in tuple_data:
            indice.append(idx)
            ins_list.append(ins)
        return indice, func(ins_list)

    return wrapper


_emitted_deprecation_warnings = set()


def deprecated(help_message: Optional[str] = None):
    """Decorator to mark a function as deprecated.

    Args:
        help_message (`Optional[str]`): An optional message to guide the user on how to
            switch to non-deprecated usage of the library.
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

    else:
        raise TypeError("Only support 1-d numpy.ndarray.")

    return mask


def wait_to_success(fn, no=False):
    while True:
        sleep(0.01)
        if (no and not fn()) or (not no and fn()):
            break


# 这个是因为在分布式文件系统中可能会发生错误，rank0下发删除成功后就运行走了，但实际的删除需要rank0的机器发送到远程文件系统再去执行，这个时候
# 在rank0那里，确实已经删除成功了，但是在远程文件系统那里这个操作还没完成，rank1读取的时候还是读取到存在这个文件；
def synchronize_safe_rm(path: Optional[Union[str, Path]]):
    if path is None:
        return
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return
    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
        _recursive_rm(path)
    wait_to_success(path.exists, no=True)


def _recursive_rm(path: Path):
    if path.is_file() or path.is_symlink():
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return
    for sub_path in list(path.iterdir()):
        _recursive_rm(sub_path)
    path.rmdir()


def synchronize_mkdir(path: Optional[Union[str, Path]]):
    """
    注意该函数是用来创建文件夹，如果需要创建一个文件，不要使用该函数；
    """
    if path is None:
        return
    if isinstance(path, str):
        path = Path(path)

    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
        path.mkdir(parents=True, exist_ok=True)

    wait_to_success(path.exists)




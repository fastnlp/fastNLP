import _pickle
import inspect
import os
import warnings
from collections import Counter
from collections import namedtuple
import torch


CheckRes = namedtuple('CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                   'varargs'], verbose=False)
def save_pickle(obj, pickle_path, file_name):
    """Save an object into a pickle file.

    :param obj: an object
    :param pickle_path: str, the directory where the pickle file is to be saved
    :param file_name: str, the name of the pickle file. In general, it should be ended by "pkl".
    """
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
        print("make dir {} before saving pickle file".format(pickle_path))
    with open(os.path.join(pickle_path, file_name), "wb") as f:
        _pickle.dump(obj, f)
    print("{} saved in {}".format(file_name, pickle_path))


def load_pickle(pickle_path, file_name):
    """Load an object from a given pickle file.

    :param pickle_path: str, the directory where the pickle file is.
    :param file_name: str, the name of the pickle file.
    :return obj: an object stored in the pickle
    """
    with open(os.path.join(pickle_path, file_name), "rb") as f:
        obj = _pickle.load(f)
    print("{} loaded from {}".format(file_name, pickle_path))
    return obj


def pickle_exist(pickle_path, pickle_name):
    """Check if a given pickle file exists in the directory.

    :param pickle_path: the directory of target pickle file
    :param pickle_name: the filename of target pickle file
    :return: True if file exists else False
    """
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    file_name = os.path.join(pickle_path, pickle_name)
    if os.path.exists(file_name):
        return True
    else:
        return False

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
    all_args = set([arg for arg in spect.args if arg!='self'])
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
    varargs = [] if spect.varargs else [arg for arg in spect.varargs]
    return CheckRes(missing=missing,
                    unused=unused,
                    duplicated=duplicated,
                    required=list(require_args),
                    all_needed=list(all_args),
                    varargs=varargs)

def get_func_signature(func):
    """

    Given a function or method, return its signature.
    For example:
    (1) function
        def func(a, b='a', *args):
            xxxx
        get_func_signature(func) # 'func(a, b='a', *args)'
    (2) method
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
        if len(signature_str)>2:
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


def _move_dict_value_to_device(*args, device:torch.device):
    """

    move data to model's device, element in *args should be dict. This is a inplace change.
    :param device: torch.device
    :param args:
    :return:
    """
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be `torch.device`, got `{type(device)}`")

    for arg in args:
        if isinstance(arg, dict):
            for key, value in arg.items():
                if isinstance(value, torch.Tensor):
                    arg[key] = value.to(device)
        else:
            raise TypeError("Only support `dict` type right now.")


class CheckError(Exception):
    """

    CheckError. Used in losses.LossBase, metrics.MetricBase.
    """
    def __init__(self, check_res:CheckRes, func_signature:str):
        errs = [f'The following problems occurred when calling {func_signature}']

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

def _check_loss_evaluate(prev_func_signature:str, func_signature:str, check_res:CheckRes,
                         output:dict, batch_y:dict, check_level=0):
    errs = []
    _unused = []
    if check_res.varargs:
        errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, "
                           f"please delete it.)")
    if check_res.missing:
        errs.append(f"\tmissing param: `{check_res.missing}`, provided with `{list(output.keys())}`"
                   f"(from output of `{prev_func_signature}`) and `{list(batch_y.keys())}`(from targets in Dataset).")
    if check_res.duplicated:
        errs.append(f"\tduplicated param: {check_res.duplicated}, delete {check_res.duplicated} in the output of "
                    f"{check_res.duplicated} or do not set {check_res.duplicated} as targets. ")
    if check_res.unused:
        _unused = [f"\tunused param: {check_res.unused}"]
        if check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs)>0:
        errs.insert(0, f'The following problems occurred when calling {func_signature}')
        raise NameError('\n'.join(errs))
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def _check_forward_error(forward_func, batch_x, check_level):
    check_res = _check_arg_dict_list(forward_func, batch_x)
    func_signature = get_func_signature(forward_func)

    errs = []
    _unused = []

    if check_res.varargs:
        errs.append(f"\tvarargs: {check_res.varargs}(Does not support pass positional arguments, please delete it)")
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}, provided with {list(batch_x.keys())}.")
    if check_res.unused:
        _unused = [f"\tunused param: {check_res.unused}"]
        if check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs)>0:
        errs.insert(0, f'The following problems occurred when calling {func_signature}')
        raise NameError('\n'.join(errs))
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)
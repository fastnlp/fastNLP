import _pickle
import inspect
import os
import warnings
from collections import Counter
from collections import namedtuple

import numpy as np
import torch

CheckRes = namedtuple('CheckRes', ['missing', 'unused', 'duplicated', 'required', 'all_needed',
                                   'varargs'])


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


def _move_dict_value_to_device(*args, device: torch.device):
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

    def __init__(self, check_res: CheckRes, func_signature: str):
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


def _check_loss_evaluate(prev_func_signature: str, func_signature: str, check_res: CheckRes,
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
            unuseds.append(f"\tunused param: {_unused_param}") # output from predict or forward

    module_name = func_signature.split('.')[0]
    if check_res.missing:
        errs.append(f"\tmissing param: {check_res.missing}")
        import re
        mapped_missing = []
        unmapped_missing = []
        input_func_map = {}
        for _miss in check_res.missing:
            if '(' in _miss:
                # if they are like 'SomeParam(assign to xxx)'
                _miss = _miss.split('(')[0]
            matches = re.findall("(?<=`)[a-zA-Z0-9]*?(?=`)", _miss)
            if len(matches) == 2:
                fun_arg, module_name = matches
                input_func_map[_miss] = fun_arg
                if fun_arg == _miss:
                    unmapped_missing.append(_miss)
                else:
                    mapped_missing.append(_miss)
            else:
                unmapped_missing.append(_miss)

        for _miss in mapped_missing:
            if _miss in dataset:
                suggestions.append(f"Set {_miss} as target.")
            else:
                _tmp = ''
                if check_res.unused:
                    _tmp = f"Check key assignment for `{input_func_map.get(_miss, _miss)}` when initialize {module_name}."
                if _tmp:
                    _tmp += f' Or provide {_miss} in DataSet or output of {prev_func_signature}.'
                else:
                    _tmp = f'Provide {_miss} in DataSet or output of {prev_func_signature}.'
                suggestions.append(_tmp)
        for _miss in unmapped_missing:
            if _miss in dataset:
                suggestions.append(f"Set {_miss} as target.")
            else:
                _tmp = ''
                if check_res.unused:
                    _tmp = f"Specify your assignment for `{input_func_map.get(_miss, _miss)}` when initialize {module_name}."
                if _tmp:
                    _tmp += f' Or provide {_miss} in DataSet or output of {prev_func_signature}.'
                else:
                    _tmp = f'Provide {_miss} in output of {prev_func_signature} or DataSet.'
                suggestions.append(_tmp)

    if check_res.duplicated:
        errs.append(f"\tduplicated param: {check_res.duplicated}.")
        suggestions.append(f"Delete {check_res.duplicated} in the output of "
                           f"{prev_func_signature} or do not set {check_res.duplicated} as targets. ")

    if len(errs)>0:
        errs.extend(unuseds)
    elif check_level == STRICT_CHECK_LEVEL:
        errs.extend(unuseds)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                if idx>0:
                    sugg_str += '\t\t\t'
                sugg_str += f'({idx+1}). {sugg}\n'
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
    func_signature = get_func_signature(forward_func)

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
        if len(errs)>0:
            errs.extend(_unused)
        elif check_level == STRICT_CHECK_LEVEL:
            errs.extend(_unused)

    if len(errs) > 0:
        errs.insert(0, f'Problems occurred when calling {func_signature}')
        sugg_str = ""
        if len(suggestions) > 1:
            for idx, sugg in enumerate(suggestions):
                sugg_str += f'({idx+1}). {sugg}'
        else:
            sugg_str += suggestions[0]
        err_str = '\n' + '\n'.join(errs) + '\n\tSuggestion: ' + sugg_str
        raise NameError(err_str)
    if _unused:
        if check_level == WARNING_CHECK_LEVEL:
            _unused_warn = _unused[0] + f' in {func_signature}.'
            warnings.warn(message=_unused_warn)


def seq_lens_to_masks(seq_lens, float=False):
    """

    Convert seq_lens to masks.
    :param seq_lens: list, np.ndarray, or torch.LongTensor, shape should all be (B,)
    :param float: if True, the return masks is in float type, otherwise it is byte.
    :return: list, np.ndarray or torch.Tensor, shape will be (B, max_length)
    """
    if isinstance(seq_lens, np.ndarray):
        assert len(np.shape(seq_lens)) == 1, f"seq_lens can only have one dimension, got {len(np.shape(seq_lens))}."
        assert seq_lens.dtype in (int, np.int32, np.int64), f"seq_lens can only be integer, not {seq_lens.dtype}."
        raise NotImplemented
    elif isinstance(seq_lens, torch.Tensor):
        assert len(seq_lens.size()) == 1, f"seq_lens can only have one dimension, got {len(seq_lens.size())==1}."
        batch_size = seq_lens.size(0)
        max_len = seq_lens.max()
        indexes = torch.arange(max_len).view(1, -1).repeat(batch_size, 1).to(seq_lens.device)
        masks = indexes.lt(seq_lens.unsqueeze(1))

        if float:
            masks = masks.float()

        return masks
    elif isinstance(seq_lens, list):
        raise NotImplemented
    else:
        raise NotImplemented


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    :param seq_len: list or torch.Tensor, the lengths of sequences in a batch.
    :param max_len: int, the maximum sequence length in a batch.
    :return mask: torch.LongTensor, [batch_size, max_len]

    """
    if not isinstance(seq_len, torch.Tensor):
        seq_len = torch.LongTensor(seq_len)
    seq_len = seq_len.view(-1, 1).long()   # [batch_size, 1]
    seq_range = torch.arange(start=0, end=max_len, dtype=torch.long, device=seq_len.device).view(1, -1) # [1, max_len]
    return torch.gt(seq_len, seq_range) # [batch_size, max_len]


class pseudo_tqdm:
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

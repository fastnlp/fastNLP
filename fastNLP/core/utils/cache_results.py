from datetime import datetime
import hashlib
import _pickle
import functools
import os
from typing import Callable, List, Any, Optional
import inspect
import ast
from collections import deque

__all__ = [
    'cache_results'
]

from fastNLP.core.log.logger import logger
from fastNLP.core.log.highlighter import ColorHighlighter


class FuncCallVisitor(ast.NodeVisitor):
    # credit to https://gist.github.com/jargnar/0946ab1d985e2b4ab776
    def __init__(self):
        self._name = deque()

    @property
    def name(self):
        return '.'.join(self._name)

    @name.deleter
    def name(self):
        self._name.clear()

    def visit_Name(self, node):
        self._name.appendleft(node.id)

    def visit_Attribute(self, node):
        try:
            self._name.appendleft(node.attr)
            self._name.appendleft(node.value.id)
        except AttributeError:
            self.generic_visit(node)


def get_func_calls(tree):
    func_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callvisitor = FuncCallVisitor()
            callvisitor.visit(node.func)
            func_calls.append(callvisitor.name)
        if isinstance(node, ast.FunctionDef):
            if not (node is tree):
                func_calls.extend(get_func_calls(node))

    return func_calls


def truncate_start_blanks(source:str)->str:
    """
    将source中的每一行按照第一行的indent删掉多余的空格

    :param source:
    :return:
    """
    lines = source.split('\n')
    num_blank = 0
    # get the top blank line
    for line in lines:
        if line:
            num_blank = len(line) - len(line.lstrip())
    new_lines = []
    for line in lines:
        i = -1
        for i in range(min(len(line), num_blank)):
            if line[i] == ' ':
                continue
            else:
                break
        line = line[i:]
        new_lines.append(line)
    return '\n'.join(new_lines)


def _get_func_and_its_called_func_source_code(func) -> List[str]:
    """
    给定一个func，返回在这个函数里面用到的所有函数的源码。

    :param callable func:
    :return:
    """
    last_frame = inspect.currentframe().f_back.f_back.f_back
    last_frame_f_local = last_frame.f_locals
    last_frame_loc = {}
    if 'loc' in last_frame_f_local:
        last_frame_loc = last_frame_f_local['loc']
    func_calls = list(set(get_func_calls(ast.parse(truncate_start_blanks(inspect.getsource(func))))))
    func_calls.sort()
    sources = []
    for _func_name in func_calls:
        try:
            if _func_name == 'cache_results':  # ignore the decorator
                continue
            if '.' in _func_name:
                _funcs = _func_name.split('.')
            else:
                _funcs = [_func_name]
            if _funcs[0] in last_frame_f_local or _funcs[0] in last_frame_loc:
                tmp = _funcs.pop(0)
                variable = last_frame_f_local.get(tmp, last_frame_loc.get(tmp))
                while len(_funcs) or variable is not None:
                    if hasattr(variable, '__class__') and not inspect.isbuiltin(variable.__class__):
                        try:
                            sources.append(inspect.getsource(variable.__class__))
                        except TypeError:
                            pass
                    if callable(variable) or inspect.isclass(variable):
                        sources.append(inspect.getsource(variable))
                    if len(_funcs):
                        tmp = _funcs.pop(0)
                        if hasattr(variable, tmp):
                            variable = getattr(variable, tmp)
                        else:
                            break
                    else:
                        variable = None
        except:
            # some failure
            pass
    del last_frame  #
    sources.append(inspect.getsource(func))
    return sources


def _prepare_cache_filepath(filepath:str):
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


class Hasher:
    def __init__(self):
        self.m = hashlib.sha1()

    def update(self, value: Any) -> None:
        if isinstance(value, str):
            value = [value]
        for x in value:
            self.m.update(x.encode('utf8'))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


def cal_fn_hash_code(fn: Optional[Callable] = None, fn_kwargs: Optional[dict] = None):
    if fn_kwargs is None:
        fn_kwargs = {}
    hasher = Hasher()
    try:
        sources = _get_func_and_its_called_func_source_code(fn)
        hasher.update(sources)
    except:
        return "can't be hashed"
    for key in sorted(fn_kwargs):
        hasher.update(key)
        try:
            hasher.update(fn_kwargs[key])
        except:
            pass
    return hasher.hexdigest()


def cache_results(_cache_fp, _refresh=False, _verbose=1, _check_hash=True):
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
    :param bool _check_hash: 如果为 True 将尝试对比修饰的函数的源码以及该函数内部调用的函数的源码的hash值。如果发现保存时的hash值
        与当前的hash值有差异，会报warning。但该warning可能出现实质上并不影响结果的误报（例如增删空白行）；且在修改不涉及源码时，虽然
        该修改对结果有影响，但无法做出warning。

    :return:
    """

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose', '_check_hash'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fn_param = kwargs.copy()
            if args:
                params = [p.name for p in inspect.signature(func).parameters.values()]
                fn_param.update(zip(params, args))
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

            if '_check_hash' in kwargs:
                check_hash = kwargs.pop('_check_hash')
            else:
                check_hash = _check_hash

            refresh_flag = True
            new_hash_code = None
            if check_hash:
                new_hash_code = cal_fn_hash_code(func, fn_param)

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    cache_filepath = os.path.abspath(cache_filepath)
                    with open(cache_filepath, 'rb') as f:
                        results = _pickle.load(f)
                        old_hash_code = results['hash']
                        save_time = results['save_time']
                        results = results['results']
                    if verbose == 1:
                        logger.info("Read cache from {} (Saved on {}).".format(cache_filepath, save_time))
                    if check_hash and old_hash_code != new_hash_code:
                        logger.warning(f"The function `{func.__name__}` is different from its last cache (Save on {save_time}). The "
                                      f"difference may caused by the sourcecode change of the functions by this function.",
                                       extra={'highlighter': ColorHighlighter('red')})
                    refresh_flag = False

            if refresh_flag:
                if new_hash_code is None:
                    new_hash_code = cal_fn_hash_code(func, fn_param)
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Cannot save None results.")
                    cache_filepath = os.path.abspath(cache_filepath)
                    _prepare_cache_filepath(cache_filepath)
                    _dict = {
                        'results': results,
                        'hash': new_hash_code,
                        'save_time': datetime.now(),
                    }
                    with open(cache_filepath, 'wb') as f:
                        _pickle.dump(_dict, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_
"""
:func:`cache_results` 函数是 **fastNLP** 中用于缓存数据的装饰器，通过该函数您可以省去调试代码过程中一些耗时过长程序
带来的时间开销。比如在加载并处理较大的数据时，每次修改训练参数都需要从头开始执行处理数据的过程，那么 :func:`cache_results`
便可以跳过这部分漫长的时间。详细的使用方法和原理请参见下面的说明。

.. warning::

    如果您发现对代码进行修改之后程序执行的结果没有变化，很有可能是这个函数的原因；届时删除掉缓存数据即可。

"""

from datetime import datetime
import hashlib
import _pickle
import functools
import os
import re
from typing import Callable, List, Any, Optional
import inspect
import ast
from collections import deque

__all__ = [
    'cache_results'
]

from fastNLP.core.log.logger import logger
from fastNLP.core.log.highlighter import ColorHighlighter
from .utils import _get_fun_msg


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
    func_source_code = inspect.getsource(func)  # 将这个函数中的 cache_results 装饰删除掉。
    for match in list(re.finditer('@cache_results\(.*\)\\n', func_source_code))[::-1]:
        func_source_code = func_source_code[:match.start()] + func_source_code[match.end():]
    sources.append(func_source_code)
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
    if fn is not None:
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


def cache_results(_cache_fp: str, _hash_param: bool = True, _refresh: bool = False, _verbose: int = 1, _check_hash: bool = True):
    r"""
    :func:`cache_results` 是 **fastNLP** 中用于缓存数据的装饰器。通过下面的例子看一下如何使用::

        import time
        import numpy as np
        from fastNLP import cache_results

        @cache_results('cache.pkl')
        def process_data(second=1):
            # 一些比较耗时的工作，比如读取数据，预处理数据等，这里用time.sleep()代替耗时
            time.sleep(second)
            return np.random.randint(10, size=(5,))

        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)

        start_time = time.time()
        print("res =",process_data())
        print(time.time() - start_time)

        start_time = time.time()
        print("res =",process_data(second=2))
        print(time.time() - start_time)

        # 输出内容如下，可以看到前两次结果相同，且第二次几乎没有花费时间。第三次由于参数变化了，所以cache的结果也就自然变化了。
        # Save cache to 2d145aeb_cache.pkl.
        # res = [5 4 9 1 8]
        # 1.0134737491607666
        # Read cache from 2d145aeb_cache.pkl (Saved on xxxx).
        # res = [5 4 9 1 8]
        # 0.0040721893310546875
        # Save cache to 0ead3093_cache.pkl.
        # res = [1 8 2 5 1]
        # 2.0086121559143066

    可以看到第二次运行的时候，只用了 0.0001s 左右，这是由于第二次运行将直接从cache.pkl这个文件读取数据，而不会经过再次预处理。
    如果在函数加上了装饰器 ``@cache_results()``，则函数会增加五个参数 ``[_cache_fp, _hash_param, _refresh, _verbose,
    _check_hash]``。上面的例子即为使用_cache_fp的情况，这五个参数不会传入到被装饰函数中，当然被装饰函数参数名也不能包含这五个名称。

    :param _cache_fp: 将返回结果缓存到什么位置;或从什么位置读取缓存。如果为 ``None`` ，cache_results 没有任何效用，除非在
        函数调用的时候传入 _cache_fp 这个参数。实际保存的文件名会受到 ``_hash_param`` 参数的影响，例如传入的名称是 **"caches/cache.pkl"**，
        实际保存的文件名会是 **"caches/{hash_param_result}_cache.pkl"**。
    :param _hash_param: 是否将传入给被装饰函数的 parameter 进行 :func:`str` 之后的 hash 结果加入到 ``_cache_fp`` 中，这样每次函数的
        parameter 改变的时候，cache 文件就自动改变了。
    :param _refresh: 强制重新生成新的 cache 。
    :param _verbose: 是否打印 cache 的信息。
    :param _check_hash: 如果为 ``True`` 将尝试对比修饰的函数的源码以及该函数内部调用的函数的源码的 hash 值。如果发现保存时的 hash 值
        与当前的 hash 值有差异，会报 warning 。但该 warning 可能出现实质上并不影响结果的误报（例如增删空白行）；且在修改不涉及源码时，虽然
        该修改对结果有影响，但无法做出 warning。
    :return:
    """

    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', "_hash_param", '_refresh', '_verbose', '_check_hash'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # fn_param = kwargs.copy()
            # if args:
            #     params = [p.name for p in inspect.signature(func).parameters.values()]
            #     fn_param.update(zip(params, args))
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

            if '_hash_param' in kwargs:
                hash_param = kwargs.pop('_hash_param')
                assert isinstance(hash_param, bool), "_hash_param can only be bool."
            else:
                hash_param = _hash_param

            if hash_param and cache_filepath is not None:  # 尝试将parameter给hash一下
                try:
                    params = dict(inspect.getcallargs(func, *args, **kwargs))
                    if inspect.ismethod(func):  # 如果是 method 的话第一个参数（一般就是 self ）就不考虑了
                        first_key = next(iter(params.items()))
                        params.pop(first_key)
                    if len(params):
                        # sort 一下防止顺序改变
                        params = {k: str(v) for k, v in sorted(params.items(), key=lambda item: item[0])}
                        param_hash = cal_fn_hash_code(None, params)[:8]
                        head, tail = os.path.split(cache_filepath)
                        cache_filepath = os.path.join(head, param_hash + '_' + tail)
                except BaseException as e:
                    logger.debug(f"Fail to add parameter hash to cache path, because of Exception:{e}")

            refresh_flag = True
            new_hash_code = None
            if check_hash:
                new_hash_code = cal_fn_hash_code(func, None)

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
                        logger.warning(f"The function {_get_fun_msg(func)} is different from its last cache (Save on {save_time}). The "
                                      f"difference may caused by the sourcecode change.",
                                       extra={'highlighter': ColorHighlighter('red')})
                    refresh_flag = False

            if refresh_flag:
                if new_hash_code is None:
                    new_hash_code = cal_fn_hash_code(func, None)
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
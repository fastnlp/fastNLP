import os
import sys
import __main__
from functools import wraps
import inspect
from inspect import ismethod
import functools
from copy import deepcopy
from io import StringIO
import time

import numpy as np

from fastNLP.envs.env import FASTNLP_GLOBAL_RANK
from fastNLP.core.drivers.utils import distributed_open_proc
from fastNLP.core.log import logger


def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


def recover_logger(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # 保存logger的状态
        handlers = [handler for handler in logger.handlers]
        level = logger.level
        res = fn(*args, **kwargs)
        logger.handlers = handlers
        logger.setLevel(level)
        return res

    return wrapper


def magic_argv_env_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        command = deepcopy(sys.argv)
        env = deepcopy(os.environ.copy())

        used_args = []
        for each_arg in sys.argv[1:]:
            if "test" not in each_arg:
                used_args.append(each_arg)

        pytest_current_test = os.environ.get('PYTEST_CURRENT_TEST')

        try:
            l_index = pytest_current_test.index("[")
            r_index = pytest_current_test.index("]")
            subtest = pytest_current_test[l_index: r_index + 1]
        except:
            subtest = ""

        if not ismethod(fn) and get_class_that_defined_method(fn) is None:
            sys.argv = [sys.argv[0], f"{os.path.abspath(sys.modules[fn.__module__].__file__)}::{fn.__name__}{subtest}"] + used_args
        else:
            sys.argv = [sys.argv[0], f"{os.path.abspath(sys.modules[fn.__module__].__file__)}::{get_class_that_defined_method(fn).__name__}::{fn.__name__}{subtest}"] + used_args

        res = fn(*args, **kwargs)
        sys.argv = deepcopy(command)
        os.environ = env

        return res
    return wrapper


class Capturing(list):
    # 用来捕获当前环境中的stdout和stderr，会将其中stderr的输出拼接在stdout的输出后面
    """
    使用例子
    with Capturing() as output:
        do_something

    assert 'xxx' in output[0]
    """
    def __init__(self, no_del=False):
        # 如果no_del为True，则不会删除_stringio，和_stringioerr
        super().__init__()
        self.no_del = no_del

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = StringIO()
        sys.stderr = self._stringioerr = StringIO()
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue() + self._stringioerr.getvalue())
        if not self.no_del:
            del self._stringio, self._stringioerr    # free up some memory
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def re_run_current_cmd_for_torch(num_procs, output_from_new_proc='ignore'):
    # Script called as `python a/b/c.py`
    if int(os.environ.get('LOCAL_RANK', '0')) == 0:
        if __main__.__spec__ is None:  # pragma: no-cover
            # pull out the commands used to run the script and resolve the abs file path
            command = sys.argv
            command[0] = os.path.abspath(command[0])
            # use the same python interpreter and actually running
            command = [sys.executable] + command
        # Script called as `python -m a.b.c`
        else:
            command = [sys.executable, "-m", __main__.__spec__._name] + sys.argv[1:]

        for rank in range(1, num_procs+1):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{rank}"
            env_copy['WOLRD_SIZE'] = f'{num_procs+1}'
            env_copy['RANK'] = f'{rank}'

            # 如果是多机，一定需要用户自己拉起，因此我们自己使用 open_subprocesses 开启的进程的 FASTNLP_GLOBAL_RANK 一定是 LOCAL_RANK；
            env_copy[FASTNLP_GLOBAL_RANK] = str(rank)

            proc = distributed_open_proc(output_from_new_proc, command, env_copy, None)

            delay = np.random.uniform(1, 5, 1)[0]
            time.sleep(delay)

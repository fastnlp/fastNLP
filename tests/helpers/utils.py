import os
import sys
import __main__
from functools import wraps, partial
from inspect import ismethod
from copy import deepcopy
from io import StringIO
import time
import signal

import numpy as np

from fastNLP.core.utils.utils import get_class_that_defined_method
from fastNLP.envs.env import FASTNLP_GLOBAL_RANK
from fastNLP.core.drivers.utils import distributed_open_proc
from fastNLP.core.log import logger


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


def magic_argv_env_context(fn=None, timeout=300):
    """
    用来在测试时包裹每一个单独的测试函数，使得 ddp 测试正确；
    会丢掉 pytest 中的 arg 参数。

    :param timeout: 表示一个测试如果经过多久还没有通过的话就主动将其 kill 掉，默认为 5 分钟，单位为秒；
    :return:
    """
    # 说明是通过 @magic_argv_env_context(timeout=600) 调用；
    if fn is None:
        return partial(magic_argv_env_context, timeout=timeout)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        command = deepcopy(sys.argv)
        env = deepcopy(os.environ.copy())

        used_args = []
        # for each_arg in sys.argv[1:]:
        #     # warning，否则 可能导致 pytest -s . 中的点混入其中，导致多卡启动的 collect tests items 不为 1
        #     if each_arg.startswith('-'):
        #         used_args.append(each_arg)

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

        def _handle_timeout(signum, frame):
            raise TimeoutError(f"\nYour test fn: {fn.__name__} has timed out.\n")

        # 恢复 logger
        handlers = [handler for handler in logger.handlers]
        formatters = [handler.formatter for handler in handlers]
        level = logger.level

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout)
        res = fn(*args, **kwargs)
        signal.alarm(0)
        sys.argv = deepcopy(command)
        os.environ = env

        for formatter, handler in zip(formatters, handlers):
            handler.setFormatter(formatter)
        logger.handlers = handlers
        logger.setLevel(level)

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

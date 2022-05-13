from typing import Union
import sys

from ....envs.imports import SUPPORT_BACKENDS
from ...log import logger

from .backend import Backend
from .torch_backend.backend import TorchBackend
from .paddle_backend.backend import PaddleBackend
from .jittor_backend.backend import JittorBackend


class AutoBackend(Backend):
    """
    不需要初始化backend的AutoBackend,能够根据get_metric时候判断输入数据类型来选择backend是什么类型的

    """

    def __init__(self, backend: Union[str, Backend, None]):
        super(AutoBackend, self).__init__()
        if backend != 'auto':
            self._convert_backend(backend)

    def _convert_backend(self, backend):
        """
        将AutoBackend转换为合适的Backend对象

        """
        if isinstance(backend, Backend):
            self.__class__ = backend.__class__
        # 如果是str，直接选择就好了
        elif backend == 'torch':
            self.__class__ = TorchBackend
        elif backend == 'paddle':
            self.__class__ = PaddleBackend
        elif backend == 'jittor':
            self.__class__ = JittorBackend
        elif backend is None:
            # 不用做任何事情就可以初始化了
            pass
        else:
            raise RuntimeError(f"We did not support `{backend}` to be used as backend for now.")
        self._specified = True

    def choose_real_backend(self, args):
        assert not self.is_specified(), "This method should not be called after backend has been specified. " \
                                        "This must be a bug, please report."
        types = []
        for arg in args:
            types.append(str(type(arg)))

        torch_types = []
        jittor_types = []
        paddle_types = []
        for type_name in types:
            if 'torch' in type_name:
                torch_types.append(type_name)
            if 'paddle' in type_name:
                paddle_types.append(type_name)
            if 'jittor' in type_name:
                jittor_types.append(type_name)

        # 根据 https://stackoverflow.com/a/3464154 ，可以通过这种方法实现切换成真实的 backend 上
        if len(torch_types) > 0 and len(jittor_types) == 0 and len(paddle_types) == 0:
            backend = 'torch'
        elif len(torch_types) == 0 and len(jittor_types) > 0 and len(paddle_types) == 0:
            backend = 'jittor'
        elif len(torch_types) == 0 and len(jittor_types) == 0 and len(paddle_types) > 0:
            backend = 'paddle'
        elif len(torch_types) == 0 and len(jittor_types) == 0 and len(paddle_types) == 0:
            backend = None
            # 尝试通过 modules 的方式自动寻找
            find_backends = []
            for backend in SUPPORT_BACKENDS:
                if backend in sys.modules:
                    find_backends.append(backend)
            if len(find_backends) == 1:
                backend = find_backends[0]
                logger.debug(f'Find backend:{backend} through sys.modules.')
            else:
                logger.debug(f'Cannot find backend through sys.modules, since find:{find_backends}.')
        else:
            types = list(set(torch_types + jittor_types + paddle_types))
            raise RuntimeError(
                f"Mixture of tensor type:{types} have been accept, please manually set backend instead of "
                f"using backend=auto.")

        self._convert_backend(backend)

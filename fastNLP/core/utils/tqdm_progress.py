__all__ = [
    'f_tqdm_progress'
]

import uuid
import sys
from ...envs.utils import _module_available, _compare_version, _get_version

from ...envs import get_global_rank
from .utils import is_notebook
from ..log import logger
if _module_available('tqdm'):
    from tqdm.autonotebook import tqdm
import operator



class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# 如果不打印的时候，使得整个 progress 没有任何意义
class DummyFTqdmProgress:
    def __getattr__(self, item):
        return DummyFTqdmProgress()

    def __call__(self, *args, **kwargs):
        # 防止用户通过 DummyFRichProgress.console.print() 这种调用
        return None

    @property
    def dummy(self)->bool:
        """
        当前对象是否是 dummy 的 tqdm 对象。

        :return:
        """
        return True


class TqdmProgress(metaclass=Singleton):
    def __init__(self):
        self.bars = {}

    def add_task(self, iterable=None, description=None, total=None, leave=False,
                 ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
                 ascii=None, visible=True, unit='it', unit_scale=False,
                 dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
                 postfix=None, unit_divisor=1000, write_bytes=None,
                 lock_args=None, nrows=None, colour=None, gui=False, **kwargs):
        """
        主要就模仿了 tqdm bar 的创建，为了和 FRichProgress 的接口尽量统一，将 desc 重名为了 description，以及 disable 专为了
        visible 。

        :param iterable:
        :param description:
        :param total:
        :param leave:
        :param ncols:
        :param mininterval:
        :param maxinterval:
        :param miniters:
        :param ascii:
        :param visible:
        :param unit:
        :param unit_scale:
        :param dynamic_ncols:
        :param smoothing:
        :param bar_format:
        :param initial:
        :param postfix:
        :param unit_divisor:
        :param write_bytes:
        :param lock_args:
        :param nrows:
        :param colour:
        :param gui:
        :param kwargs:
        :return:
        """
        if not _module_available('tqdm'):
            raise ModuleNotFoundError("Package tqdm is not installed.")
        elif not _compare_version('tqdm', operator.ge, '4.57'):
            raise RuntimeError(f"Package tqdm>=4.57 is needed, instead of {_get_version('tqdm')}.")

        from .rich_progress import f_rich_progress
        assert not f_rich_progress.not_empty(), "Cannot use tqdm before rich finish loop."

        if hasattr(self, 'orig_out_err'):
            file = self.orig_out_err[0]
        else:
            file = sys.stdout

        bar = tqdm(iterable=iterable, desc=description, total=total, leave=leave, file=file,
                 ncols=ncols, mininterval=mininterval, maxinterval=maxinterval, miniters=miniters,
                 ascii=ascii, disable=not visible, unit=unit, unit_scale=unit_scale,
                 dynamic_ncols=dynamic_ncols, smoothing=smoothing, bar_format=bar_format, initial=initial,
                 position=len(self.bars), postfix=postfix, unit_divisor=unit_divisor, write_bytes=write_bytes,
                 lock_args=lock_args, nrows=nrows, colour=colour, gui=gui, **kwargs)
        _uuid = str(uuid.uuid1())
        self.bars[_uuid] = bar
        if not hasattr(self, 'orig_out_err') and not is_notebook():
            from tqdm.contrib import DummyTqdmFile
            self.orig_out_err = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = map(DummyTqdmFile, self.orig_out_err)

        return _uuid

    def update(self, task_id:str, advance:int, refresh=True):
        self.bars[task_id].update(advance)

    def set_postfix_str(self, task_id, s, refresh=True):
        self.bars[task_id].set_postfix_str(s=s, refresh=refresh)

    def set_description_str(self, task_id, desc, refresh=True):
        self.bars[task_id].set_description_str(desc=desc, refresh=refresh)

    def destroy_task(self, task_id):
        """
        关闭 task_id 对应的 tqdm bar 。

        :param task_id:
        :return:
        """
        self.bars[task_id].close()
        self.bars.pop(task_id)
        if len(self.bars) == 0 and hasattr(self, 'orig_out_err'):
            # recover 成正常的 sys.stdout 与 sys.stderr
            sys.stdout, sys.stderr = self.orig_out_err
            delattr(self, 'orig_out_err')

    def reset(self, task_id):
        self.bars[task_id].reset()

    def print(self):
        tqdm.write('')

    def not_empty(self):
        return len(self.bars) != 0

    @property
    def dummy(self) -> bool:
        """
        当前对象是否是 dummy 的 tqdm 对象。

        :return:
        """
        return False


if ((sys.stdin and sys.stdin.isatty()) or is_notebook()) and get_global_rank() == 0:
    f_tqdm_progress = TqdmProgress()
else:
    f_tqdm_progress = DummyFTqdmProgress()
    logger.debug("Use dummy tqdm...")




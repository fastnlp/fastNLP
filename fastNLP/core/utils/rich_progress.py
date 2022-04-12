"""
该文件用于为fastNLP提供一个统一的progress bar管理，通过共用一个Task对象，trainer中的progress bar和evaluation中的progress bar才能
    不冲突

"""
import sys
from typing import Any, Union, Optional

from rich.progress import Progress, Console, GetTimeCallable, get_console, TaskID, Live
from rich.progress import ProgressColumn, TimeRemainingColumn, BarColumn, TimeElapsedColumn, TextColumn

__all__ = [
    'f_rich_progress'
]

from fastNLP.envs import get_global_rank


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# 如果不打印的时候，使得整个 progress 没有任何意义
class DummyFRichProgress:
    def __getattr__(self, item):
        return DummyFRichProgress()

    def __call__(self, *args, **kwargs):
        # 防止用户通过 DummyFRichProgress.console.print() 这种调用
        return None


class FRichProgress(Progress, metaclass=Singleton):
    """
    fastNLP 使用的 progress bar ，新增了 new_progress 函数，通过此函数即可定制 fastNLP 中所有 progress 的样式。

    """

    def new_progess(self, *columns: Union[str, ProgressColumn],
                    console: Optional[Console] = None,
                    auto_refresh: bool = True,
                    refresh_per_second: float = 10,
                    speed_estimate_period: float = 30.0,
                    transient: bool = True,
                    redirect_stdout: bool = True,
                    redirect_stderr: bool = True,
                    get_time: Optional[GetTimeCallable] = None,
                    disable: bool = False,
                    expand: bool = False):
        """
        重新初始化一个rich bar。如果columns不传入，则继续使用之前的column内容。

        :param progress:
        :return:
        """
        for task_id in self.task_ids:  # 首先移除已有的
            self.remove_task(task_id)

        assert (
                refresh_per_second is None or refresh_per_second > 0
        ), "refresh_per_second must be > 0"

        # stop previous columns
        self.stop()

        # do not change these variables
        # self._lock = RLock()
        # self._tasks: Dict[TaskID, Task] = {}
        # self._task_index: TaskID = TaskID(0)

        if len(columns) != 0:
            self.columns = columns

        self.speed_estimate_period = speed_estimate_period

        self.disable = disable
        self.expand = expand

        self.live = Live(
            console=console or get_console(),
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_renderable=self.get_renderable,
        )
        self.get_time = get_time or self.console.get_time
        self.print = self.console.print
        self.log = self.console.log

        return self

    def set_transient(self, transient: bool = True):
        """
        设置是否在bar运行结束之后不关闭

        :param transient:
        :return:
        """
        self.new_progess(transient=transient)

    def set_disable(self, flag: bool = True):
        """
        设置当前 progress bar 的状态，如果为 True ，则不会显示进度条了。

        :param flag:
        :return:
        """
        self.disable = flag

    def add_task(
            self,
            description: str,
            start: bool = True,
            total: float = 100.0,
            completed: int = 0,
            visible: bool = True,
            **fields: Any,
    ) -> TaskID:
        if self.live._started is False:
            self.start()
        post_desc = fields.pop('post_desc', '')
        return super().add_task(description=description,
                                start=start,
                                total=total,
                                completed=completed,
                                visible=visible,
                                post_desc=post_desc,
                                **fields)

    def stop_task(self, task_id: TaskID) -> None:
        if task_id in self._tasks:
            super().stop_task(task_id)

    def remove_task(self, task_id: TaskID) -> None:
        if task_id in self._tasks:
            super().remove_task(task_id)

    def destroy_task(self, task_id: TaskID):
        if task_id in self._tasks:
            super().stop_task(task_id)
            super().remove_task(task_id)

    def start(self) -> None:
        super().start()
        self.console.show_cursor(show=True)


if (sys.stdin and sys.stdin.isatty()) and get_global_rank() == 0:
    f_rich_progress = FRichProgress().new_progess(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        TextColumn("{task.fields[post_desc]}", justify="right"),
        transient=True,
        disable=False,
        speed_estimate_period=1
    )
else:
    f_rich_progress = DummyFRichProgress()


if __name__ == '__main__':
    f = DummyFRichProgress()
    f.console.print('xxx')
    f.console.print.print('xxx')
    # 测试创建
    import time

    n_steps = 10

    task_id = f_rich_progress.add_task(description='test', total=n_steps)
    for i in range(n_steps):
        f_rich_progress.update(task_id, description=f'test:{i}', advance=1, refresh=True)
        print(f"test:{i}")
        time.sleep(0.3)
    f_rich_progress.remove_task(task_id)

    # 测试一下 inner/outer
    n_steps = 5
    f_rich_progress.start()
    outer_task_id = f_rich_progress.add_task(description='Outer:', total=n_steps)
    inner_task_id = f_rich_progress.add_task(description='Inner:', total=n_steps)
    for i in range(n_steps):
        f_rich_progress.reset(inner_task_id, total=n_steps)
        f_rich_progress.update(outer_task_id, description=f'Outer:{i}', advance=1, refresh=True)
        for j in range(n_steps):
            f_rich_progress.update(inner_task_id, description=f'Inner:{j}', advance=1, refresh=True,
                                   post_desc='Loss: 0.334332323')
            print(f"Outer:{i}, Inner:{j}")
            time.sleep(0.3)

    # 测试一下修改bar
    f_rich_progress = FRichProgress().new_progess(
        BarColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        transient=True)
    n_steps = 10
    task_id = f_rich_progress.add_task(description='test', total=n_steps)
    for i in range(n_steps):
        f_rich_progress.update(task_id, description=f'test:{i}', advance=1)
        print(f"test:{i}")
        time.sleep(0.3)
    f_rich_progress.remove_task(task_id)
    f_rich_progress.stop()

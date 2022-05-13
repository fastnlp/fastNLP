"""
该文件用于为 **fastNLP** 提供一个统一的 ``progress bar`` 管理，通过共用一个``Task`` 对象， :class:`~fastNLP.core.Trainer`
中的 ``progress bar`` 和 :class:`~fastNLP.core.Evaluator` 中的 ``progress bar`` 才能不冲突
"""
import sys
from typing import Any, Union, Optional

from rich.progress import Progress, Console, GetTimeCallable, get_console, TaskID, Live, Text, ProgressSample
from rich.progress import ProgressColumn, TimeRemainingColumn, BarColumn, TimeElapsedColumn, TextColumn

__all__ = [
    'f_rich_progress'
]

from fastNLP.envs import get_global_rank
from .utils import is_notebook


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

    @property
    def dummy_rich(self)->bool:
        """
        当前对象是否是 dummy 的 rich 对象。

        :return:
        """
        return True

class FRichProgress(Progress, metaclass=Singleton):
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
            self.refresh()  # 使得bar不残留
        # 这里需要注释掉的原因是由于，在dataset多次apply的过程中会出现自动换行的问题。以前保留这个的原因应该是由于evaluate结束bar不消失。
        # if len(self._tasks) == 0:
        #     self.live.stop()

    def start(self) -> None:
        super().start()
        self.console.show_cursor(show=True)

    def update(
            self,
            task_id: TaskID,
            *,
            total: Optional[float] = None,
            completed: Optional[float] = None,
            advance: Optional[float] = None,
            description: Optional[str] = None,
            visible: Optional[bool] = None,
            refresh: bool = False,
            **fields: Any,
    ) -> None:
        """Update information associated with a task.

        Args:
            task_id (TaskID): Task id (returned by add_task).
            total (float, optional): Updates task.total if not None.
            completed (float, optional): Updates task.completed if not None.
            advance (float, optional): Add a value to task.completed if not None.
            description (str, optional): Change task description if not None.
            visible (bool, optional): Set visible flag if not None.
            refresh (bool): Force a refresh of progress information. Default is False.
            **fields (Any): Additional data fields required for rendering.
        """
        with self._lock:
            task = self._tasks[task_id]
            completed_start = task.completed

            if total is not None and total != task.total:
                task.total = total
                task._reset()
            if advance is not None:
                task.completed += advance
            if completed is not None:
                task.completed = completed
            if description is not None:
                task.description = description
            if visible is not None:
                task.visible = visible
            task.fields.update(fields)
            update_completed = task.completed - completed_start

            current_time = self.get_time()
            old_sample_time = current_time - self.speed_estimate_period
            _progress = task._progress

            popleft = _progress.popleft
            # 这里修改为至少保留一个，防止超长时间的迭代影响判断
            while len(_progress)>1 and _progress[0].timestamp < old_sample_time:
                popleft()
            if update_completed > 0:
                _progress.append(ProgressSample(current_time, update_completed))
            if task.completed >= task.total and task.finished_time is None:
                task.finished_time = task.elapsed

        if refresh:
            self.refresh()

    @property
    def dummy_rich(self) -> bool:
        """
        当前对象是否是 dummy 的 rich 对象。

        :return:
        """
        return False


class SpeedColumn(ProgressColumn):
    """
    显示 task 的速度。

    """
    def render(self, task: "Task"):
        speed = task.speed
        if speed is None:
            return Text('-- it./s', style='progress.data.speed')
        if speed > 0.1:
            return Text(str(round(speed, 2))+' it./s', style='progress.data.speed')
        else:
            return Text(str(round(1/speed, 2))+' s/it.', style='progress.data.speed')


if ((sys.stdin and sys.stdin.isatty()) or is_notebook()) and \
        get_global_rank() == 0:
    f_rich_progress = FRichProgress().new_progess(
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(),
        SpeedColumn(),
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        TextColumn("{task.fields[post_desc]}", justify="right"),
        transient=True,
        disable=False,
        speed_estimate_period=30
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

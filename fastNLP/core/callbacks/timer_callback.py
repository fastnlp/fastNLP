import time
from .callback import Callback
from ..log import logger
__all__ = ['TimerCallback']


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, f'{self.name_} timer has already been started'
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, f'{self.name_} timer is not started'
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, item):
        return item in self.timers

    def reset(self):
        for timer in self.timers.values():
            timer.reset()


class TimerCallback(Callback):
    """
    这个 callback 的作用是打印训练过程中的相关时间信息，例如训练时长、评测时长、总时长等

    """
    def __init__(self, print_every=-1, time_ndigit=3):
        """

        :param print_every: 在哪个时候打印时间信息。

            * *负数*: 表示每隔多少 epoch 结束打印一次；
            * *0*: 表示整个训练结束才打印；
            * *正数*: 每隔多少个 step 打印一次；

        :param time_ndigit: 保留多少位的小数
        """
        assert isinstance(print_every, int), "print_every must be an int number."
        self.timers = Timers()
        self.print_every = print_every
        self.time_ndigit = time_ndigit

    def on_train_begin(self, trainer):
        self.timers('total').start()
        self.timers('train').start()

    def on_fetch_data_begin(self, trainer):
        self.timers('fetch-data').start()

    def on_fetch_data_end(self, trainer):
        self.timers('fetch-data').stop()

    def on_train_batch_begin(self, trainer, batch, indices):
        self.timers('forward').start()

    def on_before_backward(self, trainer, outputs):
        self.timers('forward').stop()
        self.timers('backward').start()

    def on_after_backward(self, trainer):
        self.timers('backward').stop()

    def on_before_optimizers_step(self, trainer, optimizers):
        self.timers('optimize').start()

    def on_after_optimizers_step(self, trainer, optimizers):
        self.timers('optimize').stop()

    def on_evaluate_begin(self, trainer):
        self.timers('train').stop()
        self.timers('evaluate').start()

    def on_evaluate_end(self, trainer, results):
        self.timers('evaluate').stop()
        self.timers('train').start()

    def format_timer(self, reset=True):
        line = ''
        timers = ['fetch-data', 'forward', 'backward', 'optimize', 'evaluate', 'train', 'total']
        for timer_name in timers:
            if not timer_name in self.timers:
                continue
            timer = self.timers(timer_name)
            elapsed = round(timer.elapsed(reset=reset), self.time_ndigit)
            if elapsed != 0:
                line = line + f', {timer_name}: {elapsed}s'
        return line

    def on_train_batch_end(self, trainer):
        if self.print_every>0 and trainer.global_forward_batches % self.print_every == 0:
            line = self.format_timer()
            logger.info(f"Running {self.print_every} batches{line}")

    def on_train_epoch_end(self, trainer):
        if self.print_every < 0 and trainer.cur_epoch_idx % abs(self.print_every) == 0:
            line = self.format_timer()
            logger.info(f"Running {abs(self.print_every)} epochs{line}")

    def on_train_end(self, trainer):
        if self.print_every == 0:
            line = self.format_timer()
            logger.info(f"Training finished{line}")




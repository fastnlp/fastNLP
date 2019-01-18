import numpy as np
import random
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.dataloader import _set_worker_signal_handlers, _update_worker_pids, \
    _remove_worker_pids, _error_if_any_worker_fails
import signal
import sys
import threading
import traceback
import os
from torch._six import FileNotFoundError

from fastNLP.core.sampler import RandomSampler

class Batch(object):
    def __init__(self, dataset, batch_size, sampler=RandomSampler(), as_numpy=False, num_workers=0, pin_memory=False,
                 timeout=0.0):
        """
        Batch is an iterable object which iterates over mini-batches.

            Example::

                for batch_x, batch_y in Batch(data_set, batch_size=16, sampler=SequentialSampler()):
                    # ...

        :param DataSet dataset: a DataSet object
        :param int batch_size: the size of the batch
        :param Sampler sampler: a Sampler object
        :param bool as_numpy: If True, return Numpy array when possible. Otherwise, return torch tensors.
        :param num_workers: int, 使用多少个进程来准备数据。默认为0, 即使用主线程生成数据。 特性处于实验阶段，谨慎使用。
            如果DataSet较大，且每个batch的准备时间很短，使用多进程可能并不能提速。
        :param pin_memory: bool, 默认为False. 设置为True时，有可能可以节省tensor从cpu移动到gpu的阻塞时间。
        :param timeout: float, 大于0的数，只有在num_workers>0时才有用。超过该时间仍然没有获取到一个batch则报错，可以用于
            检测是否出现了batch产生阻塞的情况。
        """

        if num_workers < 0:
            raise ValueError('num_workers option cannot be negative; '
                             'use num_workers=0 to disable multiprocessing.')
        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.as_numpy = as_numpy
        self.num_batches = len(dataset) // batch_size + int(len(dataset) % batch_size != 0)
        self.cur_batch_indices = None

    def __iter__(self):
        # TODO 现在多线程的情况下每个循环都会重新创建多进程，开销可能有点大。可以考虑直接复用iterator.
        return _DataLoaderIter(self)

    def __len__(self):
        return self.num_batches

    def get_batch_indices(self):
        return self.cur_batch_indices

def to_tensor(batch, dtype):
    try:
        if dtype in (int, np.int8, np.int16, np.int32, np.int64):
            batch = torch.LongTensor(batch)
        if dtype in (float, np.float32, np.float64):
            batch = torch.FloatTensor(batch)
    except:
        pass
    return batch


"""
由于多进程涉及到大量问题，包括系统、安全关闭进程等。所以这里直接从pytorch的官方版本修改DataLoader实现多进程加速
"""

IS_WINDOWS = sys.platform == "win32"
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


_use_shared_memory = False
r"""Whether to use shared memory in default_collate"""

MANAGER_STATUS_CHECK_INTERVAL = 5.0

if IS_WINDOWS:
    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())

        def is_alive(self):
            # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
            return self.kernel32.WaitForSingleObject(self.manager_handle, 0) != 0
else:
    class ManagerWatchdog(object):
        def __init__(self):
            self.manager_pid = os.getppid()

        def is_alive(self):
            return os.getppid() == self.manager_pid


def _worker_loop(dataset, index_queue, data_queue, seed, worker_id, as_numpy):
    # 产生数据的循环
    global _use_shared_memory
    _use_shared_memory = True

    # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
    # module's handlers are executed after Python returns from C low-level
    # handlers, likely when the same fatal signal happened again already.
    # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
    _set_worker_signal_handlers()

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)

    watchdog = ManagerWatchdog()

    while True:
        try:
            # 获取当前batch计数，当前batch的indexes
            r = index_queue.get(timeout=MANAGER_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            if watchdog.is_alive():
                continue
            else:
                break
        if r is None:
            break
        idx, batch_indices = r
        try:
            # 获取相应的batch数据。这里需要修改为从dataset中取出数据并且完成padding
            samples = _get_batch_from_dataset(dataset, batch_indices, as_numpy)
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info()), batch_indices))
        else:
            data_queue.put((idx, samples, batch_indices))
            del samples

def _get_batch_from_dataset(dataset, indices, as_numpy):
    """
    给定indices，从DataSet中取出(batch_x, batch_y). 数据从这里产生后，若没有pin_memory, 则直接传递给Trainer了，如果存在
        pin_memory还会经过一道pin_memory()的处理
    :param dataset: fastNLP.DataSet对象
    :param indices: List[int], index
    :param as_numpy: bool, 是否只是转换为numpy
    :return: (batch_x, batch_y)
    """
    batch_x, batch_y = {}, {}
    for field_name, field in dataset.get_all_fields().items():
        if field.is_target or field.is_input:
            batch = field.get(indices)
            if not as_numpy and field.padder is not None:
                batch = to_tensor(batch, field.dtype)
            if field.is_target:
                batch_y[field_name] = batch
            if field.is_input:
                batch_x[field_name] = batch

    return batch_x, batch_y


def _worker_manager_loop(in_queue, out_queue, done_event, pin_memory, device_id):
    # 将数据送入到指定的query中. 即如果需要pin_memory, 则
    if pin_memory:
        torch.cuda.set_device(device_id)

    while True:
        try:
            r = in_queue.get()
        except Exception:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch, batch_indices = r
        try:
            if pin_memory:
                batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info()), batch_indices))
        else:
            out_queue.put((idx, batch, batch_indices))


def pin_memory_batch(batchs):
    """

    :param batchs: (batch_x, batch_y)
    :return: (batch_x, batch_y)
    """
    for batch_dict in batchs:
        for field_name, batch in batch_dict.items():
            if isinstance(batch, torch.Tensor):
                batch_dict[field_name] = batch.pin_memory()
    return batchs


_SIGCHLD_handler_set = False
r"""Whether SIGCHLD handler is set for DataLoader worker failures. Only one
handler needs to be set for all DataLoaders in a process."""


def _set_SIGCHLD_handler():
    # Windows doesn't support SIGCHLD handler
    if sys.platform == 'win32':
        return
    # can't set signal in child threads
    if not isinstance(threading.current_thread(), threading._MainThread):
        return
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    previous_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(previous_handler):
        previous_handler = None

    def handler(signum, frame):
        # This following call uses `waitid` with WNOHANG from C side. Therefore,
        # Python can still get and update the process status successfully.
        _error_if_any_worker_fails()
        if previous_handler is not None:
            previous_handler(signum, frame)

    signal.signal(signal.SIGCHLD, handler)
    _SIGCHLD_handler_set = True


class _DataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, batcher):
        self.batcher = batcher
        self.dataset = batcher.dataset
        self.sampler = batcher.sampler
        self.as_numpy = batcher.as_numpy
        self.batch_size = batcher.batch_size
        self.num_workers = batcher.num_workers
        self.pin_memory = batcher.pin_memory and torch.cuda.is_available()
        self.timeout = batcher.timeout
        self.done_event = threading.Event()
        self.curidx = 0
        self.idx_list = self.sampler(self.dataset)

        # self.sample_iter一次返回一个index. 可以通过其他方式替代

        base_seed = torch.LongTensor(1).random_().item()

        if self.num_workers > 0:
            # 每个worker建立一个index queue
            self.index_queues = [multiprocessing.Queue() for _ in range(self.num_workers)]
            self.worker_queue_idx = 0
            # 存放获取到的batch
            self.worker_result_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            # 这里会将batch的数据输送到self.worker_result_queue中，但是还没有送入到device中
            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queues[i],
                          self.worker_result_queue, base_seed + i,  i, self.as_numpy))
                for i in range(self.num_workers)]

            # self.data_queue取数据就行。如果有pin_memory的话，会把数据放到另一个queue
            if self.pin_memory or self.timeout > 0:
                self.data_queue = queue.Queue()
                if self.pin_memory:
                    maybe_device_id = torch.cuda.current_device()
                else:
                    # do not initialize cuda context if not necessary
                    maybe_device_id = None
                self.worker_manager_thread = threading.Thread(
                    target=_worker_manager_loop,
                    args=(self.worker_result_queue, self.data_queue, self.done_event, self.pin_memory,
                          maybe_device_id))
                self.worker_manager_thread.daemon = True
                self.worker_manager_thread.start()
            else:
                self.data_queue = self.worker_result_queue

            # worker们开始工作
            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
            _set_SIGCHLD_handler()
            self.worker_pids_set = True

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def _get_batch(self):
        if self.timeout > 0:
            try:
                return self.data_queue.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
        else:
            return self.data_queue.get()

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            if self.curidx >= len(self.idx_list):
                raise StopIteration
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            # 直接从数据集中采集数据即可
            indices = self.idx_list[self.curidx:endidx]
            self.batcher.cur_batch_indices = indices
            batch_x, batch_y = _get_batch_from_dataset(dataset=self.dataset, indices=indices,
                                                       as_numpy=self.as_numpy)
            if self.pin_memory:
                batch_x, batch_y = pin_memory_batch((batch_x, batch_y))
            self.curidx = endidx
            return batch_x, batch_y

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        # 如果生成的数据为0了，则停止
        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch, batch_indices = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            self.batcher.cur_batch_indices = batch_indices
            return self._process_next_batch(batch)

    def __iter__(self):
        self.curidx = 0

        return self

    def _put_indices(self):
        # 向采集数据的index queue中放入index
        assert self.batches_outstanding < 2 * self.num_workers
        if self.curidx >= len(self.idx_list):
            indices = None
        else:
            endidx = min(self.curidx + self.batch_size, len(self.idx_list))
            # 直接从数据集中采集数据即可
            indices = self.idx_list[self.curidx:endidx]
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.curidx = endidx
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        # 只是提醒生成下一个batch indice数据
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("_DataLoaderIter cannot be pickled")

    def _shutdown_workers(self):
        try:
            if not self.shutdown:
                self.shutdown = True
                self.done_event.set()
                for q in self.index_queues:
                    q.put(None)
                # if some workers are waiting to put, make place for them
                try:
                    while not self.worker_result_queue.empty():
                        self.worker_result_queue.get()
                except (FileNotFoundError, ImportError):
                    # Many weird errors can happen here due to Python
                    # shutting down. These are more like obscure Python bugs.
                    # FileNotFoundError can happen when we rebuild the fd
                    # fetched from the queue but the socket is already closed
                    # from the worker side.
                    # ImportError can happen when the unpickler loads the
                    # resource from `get`.
                    pass
                # done_event should be sufficient to exit worker_manager_thread,
                # but be safe here and put another None
                self.worker_result_queue.put(None)
        finally:
            # removes pids no matter what
            if self.worker_pids_set:
                _remove_worker_pids(id(self))
                self.worker_pids_set = False

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

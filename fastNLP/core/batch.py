r"""
batch 模块实现了 fastNLP 所需的 :class:`~fastNLP.core.batch.DataSetIter` 类。

"""
__all__ = [
    "BatchIter",
    "DataSetIter",
    "TorchLoaderIter",
]

import atexit
import abc

from numbers import Number
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict

from .dataset import DataSet
from .sampler import SequentialSampler, Sampler
from ._logger import logger


_python_is_exit = False


def _set_python_is_exit():
    global _python_is_exit
    _python_is_exit = True


atexit.register(_set_python_is_exit)


def _pad(batch_dict, dataset, as_numpy):
    result = {}
    for n, vlist in batch_dict.items():
        f = dataset.field_arrays[n]
        if f.padder is None:
            result[n] = np.array(vlist)
        else:
            res = f.pad(vlist)
            if not as_numpy:
                res, _ = _to_tensor(res, field_dtype=f.dtype)
            result[n] = res

    return result


class DataSetGetter:
    r"""
    传递给torch.utils.data.DataLoader获取数据，DataLoder会传入int的idx获取数据(调用这里的__getitem__()函数)。
    """
    def __init__(self, dataset: DataSet, as_numpy=False):
        self.dataset = dataset
        self.as_numpy = as_numpy
        self.idx_list = list(range(len(dataset)))

        self.x_names = {n for n, f in dataset.get_all_fields().items() if f.is_input}
        self.y_names = {n for n, f in dataset.get_all_fields().items() if f.is_target}

    def __getitem__(self, idx: int):
        # mapping idx to sampled idx
        idx = self.idx_list[idx]
        ins = self.dataset[idx]
        return idx, ins

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, ins_list: list):
        r"""

        :param batch: [[idx1, x_dict1, y_dict1], [idx2, x_dict2, y_dict2], [xx, xx, xx]]
        :return:
        """
        indices = []
        sin_x, sin_y = defaultdict(list), defaultdict(list)
        # 收集需要关注的field的数据
        for idx, ins in ins_list:
            indices.append(idx)
            for n, v in ins.items():
                if n in self.x_names:
                    sin_x[n].append(v)
                if n in self.y_names:
                    sin_y[n].append(v)
        # 根据情况，进行pad
        sin_x = _pad(sin_x, dataset=self.dataset, as_numpy=self.as_numpy)
        sin_y = _pad(sin_y, dataset=self.dataset, as_numpy=self.as_numpy)

        if not self.dataset.collater.is_empty():
            bx, by = self.dataset._collate_batch(ins_list)
            sin_x.update(bx)
            sin_y.update(by)

        return indices, sin_x, sin_y

    def __getattr__(self, item):
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        else:
            raise AttributeError("'DataSetGetter' object has no attribute '{}'".format(item))


class SamplerAdapter(torch.utils.data.Sampler):
    r"""
    用于传入torch.utils.data.DataLoader中，DataLoader会调用__iter__()方法获取index(一次只取一个int)

    """
    def __init__(self, sampler, dataset):
        super().__init__(dataset)
        self.sampler = sampler
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.sampler(self.dataset))


class BatchIter:
    r"""
    Trainer用于迭代数据的类。继承该类，并实现get_num_batches(), get_batch_indices(), num_batches(), __iter__()方法以及dataset属性。

    """
    def __init__(self, dataset, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=None,
                 batch_sampler=None):
        if isinstance(sampler, Sampler):  # 如果时fastNLP的sampler需要adapt一下
            sampler = SamplerAdapter(sampler=sampler or SequentialSampler(), dataset=dataset)
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        # DataLoader的collate_fn输入是List[]，里面的元素是dataset[index]返回的结果
        if collate_fn is None:
            # pytoch <= 1.1 中不能设置collate_fn=None
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn,
                batch_sampler=batch_sampler)
        else:
            self.dataiter = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=self.sampler,
                collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, drop_last=drop_last,
                timeout=timeout, worker_init_fn=worker_init_fn,
                batch_sampler=batch_sampler)

        # 以sampler的数量为准，因为DistributedSampler的时候每个进程上并不是所有的数据都用上了
        if self.batch_sampler is None:
            self._num_batches = self.get_num_batches(len(self.dataiter.sampler), batch_size, drop_last)
        else:
            self._num_batches = len(self.batch_sampler)
        self.batch_size = batch_size
        self.cur_batch_indices = None

    @property
    def num_batches(self):
        return self._num_batches

    @num_batches.setter
    def num_batches(self, value):
        self._num_batches = value

    def init_iter(self):
        pass

    @staticmethod
    def get_num_batches(num_samples, batch_size, drop_last):
        r"""
        计算batch的数量。用于前端显示进度

        :param int num_samples:
        :param int batch_size:
        :param bool drop_last: 如果最后一个batch没有batch_size这么多，是否就丢掉。
        :return:
        """
        num_batches = num_samples // batch_size
        if not drop_last and (num_samples % batch_size > 0):
            num_batches += 1
        return num_batches

    def get_batch_indices(self):
        r"""
        获取最近输出的batch的index。用于溯源当前batch的数据

        :return:
        """
        return self.cur_batch_indices

    def __len__(self):
        return self.num_batches

    @property
    def dataset(self):
        r"""
        获取正在参与iterate的dataset

        :return:
        """
        return self.dataiter.dataset

    @abc.abstractmethod
    def __iter__(self):
        r"""
        用于实际数据循环的类，返回值需要为两个dict, 第一个dict中的内容会认为是input, 第二个dict中的内容会认为是target

        :return:
        """
        raise NotImplemented


class DataSetIter(BatchIter):
    r"""
    DataSetIter 用于从 `DataSet` 中按一定的顺序, 依次按 ``batch_size`` 的大小将数据取出，通过使用DataSetIter，可以不需要考虑
        输入的padding(由DataSet中每列的Padder决定了)以及不需要考虑将数据转为tensor。
    组成 `x` 和 `y`::

        batch = DataSetIter(data_set, batch_size=16, sampler=SequentialSampler())
        num_batch = len(batch)
        for batch_x, batch_y in batch:
            # do stuff ...

    """
    def __init__(self, dataset, batch_size=1, sampler=None, as_numpy=False, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, batch_sampler=None):
        r"""
        
        :param dataset: :class:`~fastNLP.DataSet` 对象, 数据集
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.
    
            Default: ``None``
        :param bool as_numpy: 若为 ``True`` , 输出batch为 numpy.array. 否则为 :class:`torch.Tensor`.

            Default: ``False``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param batch_sampler: 当每次batch取出的数据数量不一致时，可以使用该sampler。batch_sampler每次iter应该输出一个list的index。
            当batch_sampler不为None时，参数batch_size, sampler, drop_last会被忽略。
        """
        assert isinstance(dataset, DataSet)
        dataset = DataSetGetter(dataset, as_numpy)
        collate_fn = dataset.collate_fn
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False
        super().__init__(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler
        )

    def __iter__(self):
        self.init_iter()
        for indices, batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = indices
            yield batch_x, batch_y


class TorchLoaderIter(BatchIter):
    r"""
    与DataSetIter类似，但可以用于非fastNLP的数据容器对象，以及可以实现完全自定义的生成batch的方式，然后与Trainer，Tester可以实现
        与DataSetIter一样的对接。
    需要保证传入的数据容器实现了实现了以下的方法

    Example::

        import random
        from fastNLP import TorchLoaderIter
        import torch
        class UdfDataSet:
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):  # 必须实现的方法，输入参数是一个int，范围为[0, len(self))
                x = [random.random() for _ in range(3)]
                y = random.random()
                return x,y

            def __len__(self):  # 需要实现该方法返回值需要是一个int数据
                return self.num_samples

        # 需要实现collact_fn将数据转换为tensor
        def collate_fn(data_list):
            # [(x1,y1), (x2,y2), ...], 这里的输入实际上是将UdfDataSet的__getitem__输入结合为list
            xs, ys = [], []
            for l in data_list:
                x, y = l
                xs.append(x)
                ys.append(y)
            # 不需要转移到gpu，Trainer或Tester会将其转移到model所在的device
            x,y = torch.FloatTensor(xs), torch.FloatTensor(ys)
            return {'x':x, 'y':y}, {'y':y}  # 第一个dict中内容类似于DataSet中的input列，第二个dict的内容类似于target列

        udf_dataset = UdfDataSet(10)
        dataset = TorchLoaderIter(udf_dataset, collate_fn=collate_fn)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 1)
            def forward(self, x, y):
                return {'loss':torch.pow(self.fc(x).squeeze(-1)-y, 2).sum()}
            def predict(self, x):
                return {'pred':self.fc(x).squeeze(0)}
        model = Model()
        trainer = Trainer(train_data=dataset, model=model, loss=None, print_every=2, dev_data=dataset,
                          metrics=AccuracyMetric(target='y'), use_tqdm=False)
        trainer.train(load_best_model=False)

    除此之外，还可以通过该方法实现OnTheFly的训练，如下面的代码所示

    Example::

        import tempfile
        import random
        import torch
        tmp_file_handler, tmp_file_path = tempfile.mkstemp(text=True)
        try:
            num_samples, data = 10, []
            for _ in range(num_samples):
                x, y = [random.random() for _ in range(3)], random.random()
                data.append(x + [y])
            with open(tmp_file_path, 'w') as f:
                for d in data:
                    f.write(' '.join(map(str, d)) + '\n')

            class FileDataSet:
                def __init__(self, tmp_file):
                    num_samples = 0
                    line_pos = [0]  # 对应idx是某一行对应的位置
                    self.tmp_file_handler = open(tmp_file, 'r', encoding='utf-8')
                    line = self.tmp_file_handler.readline()
                    while line:
                        if line.strip():
                            num_samples += 1
                            line_pos.append(self.tmp_file_handler.tell())
                        line = self.tmp_file_handler.readline()
                    self.tmp_file_handler.seek(0)
                    self.num_samples = num_samples
                    self.line_pos = line_pos

                def __getitem__(self, idx):
                    line_start, line_end = self.line_pos[idx], self.line_pos[idx + 1]
                    self.tmp_file_handler.seek(line_start)
                    line = self.tmp_file_handler.read(line_end - line_start).strip()
                    values = list(map(float, line.split()))
                    x, y = values[:3], values[-1]
                    return x, y

                def __len__(self):
                    return self.num_samples

            def collate_fn(data_list):
                # [(x1,y1), (x2,y2), ...], 这里的输入实际上是将UdfDataSet的__getitem__输入结合为list
                xs, ys = [], []
                for l in data_list:
                    x, y = l
                    xs.append(x)
                    ys.append(y)
                x, y = torch.FloatTensor(xs), torch.FloatTensor(ys)
                return {'x': x, 'y': y}, {'y': y}  # 第一个dict中内容类似于DataSet中的input列，第二个dict的内容类似于target列

            file_data = FileDataSet(tmp_file_path)
            dataset = TorchLoaderIter(file_data, collate_fn=collate_fn)

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(3, 1)

                def forward(self, x, y):
                    return {'loss': torch.pow(self.fc(x).squeeze(-1) - y, 2).sum()}

                def predict(self, x):
                    return {'pred': self.fc(x).squeeze(0)}

            model = Model()
            trainer = Trainer(train_data=dataset, model=model, loss=None, print_every=2, dev_data=dataset,
                              metrics=AccuracyMetric(target='y'), use_tqdm=False, n_epochs=2)
            trainer.train(load_best_model=False)

        finally:
            import os
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    
    """
    def __init__(self, dataset, collate_fn, batch_size=1, sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None,
                 batch_sampler=None):
        r"""

        :param dataset: 实现了__getitem__和__len__方法的数据容器。
        :param callable collate_fn: 用于将样本组合成batch的函数。输入为[dataset[idx1], dataset[idx2], ...], 即dataset中
            __getitem__返回值组成的list，返回值必须为两个dict，其中第一个dict会被认为是input，第二个dict中的内容被认为是target。
            需要转换为tensor的数据，需要在collate_fn中转化，但不需要转移到对应device。
        :param int batch_size: 取出的batch大小
        :param sampler: 规定使用的 :class:`~fastNLP.Sampler` 方式. 若为 ``None`` , 使用 :class:`~fastNLP.SequentialSampler`.
            Default: ``None``
        :param int num_workers: 使用多少个进程来预处理数据
        :param bool pin_memory: 是否将产生的tensor使用pin memory, 可能会加快速度。
        :param bool drop_last: 如果最后一个batch没有batch_size这么多sample，就扔掉最后一个
        :param timeout: 生成一个batch的timeout值
        :param worker_init_fn: 在每个worker启动时调用该函数，会传入一个值，该值是worker的index。
        :param batch_sampler: 当每次batch取出的数据数量不一致时，可以使用该sampler。batch_sampler每次iter应该输出一个list的index。
            当batch_sampler不为None时，参数batch_size, sampler, drop_last会被忽略。
        """
        assert len(dataset) > 0
        assert collate_fn is not None, "You must pass collate_fn to pad the batch."
        if batch_sampler is not None:
            batch_size = 1
            sampler = None
            drop_last = False

        super().__init__(
            dataset=dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
            collate_fn=collate_fn, batch_sampler=batch_sampler
        )

    def __iter__(self):
        self.init_iter()
        for batch_x, batch_y in self.dataiter:
            self.cur_batch_indices = None
            yield batch_x, batch_y


def _to_tensor(batch, field_dtype):
    r"""

    :param batch: np.array()
    :param field_dtype: 数据类型
    :return: batch, flag. 如果传入的数据支持转为tensor，返回的batch就是tensor，且flag为True；如果传入的数据不支持转为tensor，
        返回的batch就是原来的数据，且flag为False
    """
    try:
        if field_dtype is not None and isinstance(field_dtype, type)\
                and issubclass(field_dtype, Number) \
                and not isinstance(batch, torch.Tensor):
            new_batch = torch.as_tensor(batch)
            flag = True
        else:
            new_batch = batch
            flag = False
        if torch.is_tensor(new_batch):
            if 'float' in new_batch.dtype.__repr__():
                new_batch = new_batch.float()
            elif 'int' in new_batch.dtype.__repr__():
                new_batch = new_batch.long()
        return new_batch, flag
    except Exception as e:
        raise e

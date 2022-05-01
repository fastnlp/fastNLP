__all__ = [
    'TorchDataLoader',
    'prepare_torch_dataloader'
]

from typing import Optional, Callable, Sequence, List, Union, Tuple, Dict, Mapping

from fastNLP.core.dataset import DataSet
from fastNLP.core.collators import AutoCollator
from fastNLP.core.collators.collator import _MultiCollator
from fastNLP.core.utils.utils import indice_collate_wrapper
from fastNLP.io.data_bundle import DataBundle
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, UnrepeatedSampler

if _NEED_IMPORT_TORCH:
    from torch.utils.data import DataLoader, Sampler
    from torch.utils.data._utils.collate import default_collate
else:
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader


class _FDataSet:
    """
    对Dataset的封装，主要是修改dataset的__getitem__函数，增加返回下标idx，值得注意的是dataset需要实现__getattribute__函数才能在_FDataset
    中调用dataset的方法
    """

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __getitem__(self, item: Union[int, list]) -> Tuple:
        return (item, self.dataset[item])

    def __getattr__(self, item):
        try:
            return self.dataset.__getattribute__(item)
        except AttributeError as e:
            raise e

    def __len__(self) -> int:
        return len(self.dataset)


class TorchDataLoader(DataLoader):
    """
    提供给使用pytorch框架的DataLoader函数，若是配套使用FastNLP的dataset则可以自动使用AutoCollate函数对数据进行自动padding操作，用户也可以通过
    提供的方法调节设置collate_fn的若干参数。
    """

    def __init__(self, dataset, batch_size: int = 1,
                 shuffle: bool = False, sampler: Union["Sampler[int]", ReproducibleSampler, UnrepeatedSampler] = None,
                 batch_sampler: Union["Sampler[Sequence[int]]", ReproducibleBatchSampler] = None,
                 num_workers: int = 0, collate_fn: Optional[Callable] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                 multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                 persistent_workers: bool = False, as_numpy: bool = False, **kwargs) -> None:
        """

        :param dataset: 实现了__getitem__和__len__的数据容器
        :param batch_size: 批次大小，当batch_sampler为None生效
        :param shuffle: 是否打乱数据集
        :param sampler: sampler实例化对象
        :param batch_sampler: batch_sampler实例化对象，其能迭代返回一个list的index数据
        :param num_workers: 进程的数量，当num_worker=0时不开启多进程
        :param collate_fn: 对取得到的数据进行打包的callable函数。[None, auto, callable]
        :param pin_memory:
        :param drop_last: 是否去掉最后一个不符合batch_size的数据
        :param timeout:
        :param worker_init_fn:
        :param multiprocessing_context:
        :param generator:
        :param prefetch_factor:
        :param persistent_workers:
        :param as_numpy: 返回数据是否设置为numpy类型，否则为torch.tensor类型
        """
        if not isinstance(dataset, _FDataSet):
            dataset = _FDataSet(dataset)

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                         batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=None,
                         pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context, generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers)
        if isinstance(dataset.dataset, DataSet):  # 使用了 fastnlp dataset
            self._collate_fn = dataset.dataset.get_collator()
            self._collate_fn.set_as_numpy(as_numpy)
            if collate_fn is not None and collate_fn is not default_collate:
                # 防止ddp重新初始化时候将torch dataloader的默认collate加进来
                self._collate_fn.add_collator(collate_fn)
        else:
            self._collate_fn = _MultiCollator(collate_fn)

        self.cur_indices_batch = None
        self.as_numpy = as_numpy

    def __getattr__(self, item):
        """
        为FDataLoader提供dataset的方法和属性，实现该方法后，用户可以在FDataLoader实例化后使用apply等dataset的方法

        :param item:
        :return:
        """
        try:
            return self.dataset.__getattr__(item)
        except AttributeError as e:
            raise e

    def __iter__(self):
        # 如果没有auto_collator 也没有自定义collate_fn， 那么此时采用dataloader自带的collate_fn， 将数据打包即可。
        if len(self._collate_fn.get_collators()) == 0:
            self._collate_fn.add_collator(self.collate_fn)
        self.collate_fn = indice_collate_wrapper(self._collate_fn)
        for indices, data in super().__iter__():
            self.cur_batch_indices = indices
            yield data

    def set_pad_val(self, *field_names, val: Optional[int] = 0) -> None:
        """
        设置每个field_name的padding值，默认为0，只有当autocollate存在时该方法有效， 若没有则会添加auto_collator函数
        当val=None时，意味着给定的field_names都不需要尝试padding

        :param field_names:
        :param val: padding值，默认为0
        :return:
        """
        flag = False
        for collator in self._collate_fn.get_collators():
            if isinstance(collator, AutoCollator):
                flag = True
                break
        if flag is False:
            self._collate_fn.add_collator(AutoCollator(self.as_numpy))
        for field_name in field_names:
            self._collate_fn.set_pad_val(field_name, val=val)

    def set_input(self, *field_names) -> None:
        """
        被设置为inputs的field_names，会输入到AutoCollator中，未被设置默认过滤掉

        :param field_names:
        :return:
        """
        flag = False
        for collator in self._collate_fn.get_collators():
            if isinstance(collator, AutoCollator):
                flag = True
                break
        if flag is False:
            self._collate_fn.add_collator(AutoCollator(self.as_numpy))
        self._collate_fn.set_input(*field_names)

    def set_collator(self, collator: Callable) -> None:
        """
        设置collate_fn函数，调用该函数后覆盖当前所有的collate_fn，包括Auto_Collate

        :param collator: 用户自定义的Callable函数
        :return:
        """
        self._collate_fn = _MultiCollator(collator)

    def add_collator(self, collator) -> None:
        """
        添加collate_fn函数，调用该函数后会将其添加到已有的collate_fn后面

        :param collator:
        :return:
        """
        self._collate_fn.add_collator(collator)

    def get_batch_indices(self) -> List[int]:
        """
        获取当前数据的idx

        :return:
        """
        return self.cur_batch_indices

    def set_pad(self):
        pass

    def set_ignore(self):
        pass

    def set_backend(self):
        pass



def prepare_torch_dataloader(ds_or_db: Union[DataSet, DataBundle, Sequence[DataSet], Mapping[str, DataSet]],
                             batch_size: int = 1,
                             shuffle: bool = False, sampler: Optional["Sampler[int]"] = None,
                             batch_sampler: Optional["Sampler[Sequence[int]]"] = None,
                             num_workers: int = 0, collate_fn: Optional[Callable] = None,
                             pin_memory: bool = False, drop_last: bool = False,
                             timeout: float = 0, worker_init_fn: Optional[Callable] = None,
                             multiprocessing_context=None, generator=None, prefetch_factor: int = 2,
                             persistent_workers: bool = False, non_train_sampler: Optional["Sampler[int]"] = None,
                             non_train_batch_size: int = 16, as_numpy: bool = False,
                             input_fields: Union[List, str, None] = None) \
        -> Union[TorchDataLoader, Dict[str, TorchDataLoader], Sequence[TorchDataLoader]]:
    """
    传入dataset或者data_bundle后，将其处理返回相对应的FdataLoader实例化对象

    :param input_fields:
    :param ds_or_db: dataset或者data_bundle
    :param batch_size: 批次大小，当batch_sampler为None生效
    :param shuffle: 是否打乱数据集
    :param sampler: sampler实例化对象
    :param batch_sampler: batch_sampler实例化对象，其能迭代返回一个list的index数据
    :param num_workers: 进程的数量，当num_worker=0时不开启多进程
    :param collate_fn: 对取得到的数据进行打包的callable函数
    :param pin_memory:
    :param drop_last: 是否去掉最后一个不符合batch_size的数据
    :param timeout:
    :param worker_init_fn:
    :param multiprocessing_context:
    :param generator:
    :param prefetch_factor:
    :param persistent_workers:
    :param non_train_sampler: 非 'train' 数据使用的 Sampler, 以及Sequence的第二个以上的ds使用的 Sampler
    :param non_train_batch_size:
    :param as_numpy: 返回数据是否设置为numpy类型，否则根据情况设置为 torch.tensor 类型。
    """
    # TODO dict, sequence情况下需要提供
    if isinstance(input_fields, str):
        input_fields = [input_fields]

    if isinstance(ds_or_db, DataSet):
        dl = TorchDataLoader(dataset=ds_or_db, batch_size=batch_size,
                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                             num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                             drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                             multiprocessing_context=multiprocessing_context, generator=generator,
                             prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                             as_numpy=as_numpy)
        if input_fields:
            dl.set_input(*input_fields)
        return dl

    elif isinstance(ds_or_db, DataBundle):
        dl_bundle = {}
        for name, ds in ds_or_db.iter_datasets():
            if 'train' in name:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=batch_size,
                                                  shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  as_numpy=as_numpy)
            else:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=non_train_batch_size,
                                                  shuffle=shuffle, sampler=non_train_sampler,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  as_numpy=as_numpy)
            if input_fields:
                dl_bundle[name].set_input(*input_fields)
        return dl_bundle

    elif isinstance(ds_or_db, Sequence):
        dl_bundle = []
        for idx, ds in enumerate(ds_or_db):
            if idx == 0:
                dl_bundle.append(
                    TorchDataLoader(dataset=ds, batch_size=batch_size,
                                    shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                    drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                    multiprocessing_context=multiprocessing_context, generator=generator,
                                    prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                    as_numpy=as_numpy)
                )
            else:
                dl_bundle.append(
                    TorchDataLoader(dataset=ds, batch_size=batch_size,
                                    shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                    drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                    multiprocessing_context=multiprocessing_context, generator=generator,
                                    prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                    as_numpy=as_numpy)
                )
        if input_fields:
            for dl in dl_bundle:
                dl.set_input(*input_fields)
        return dl_bundle

    elif isinstance(ds_or_db, Mapping):
        dl_bundle = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=batch_size,
                                                  shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  as_numpy=as_numpy)
            else:
                dl_bundle[name] = TorchDataLoader(dataset=ds, batch_size=non_train_batch_size,
                                                  shuffle=shuffle, sampler=non_train_sampler,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory,
                                                  drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers,
                                                  as_numpy=as_numpy)

            if input_fields:
                dl_bundle[name].set_input(*input_fields)

        return dl_bundle
    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or sequence or mapping!")

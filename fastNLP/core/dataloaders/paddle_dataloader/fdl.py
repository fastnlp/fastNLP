__all__ = [
    'PaddleDataLoader',
    'prepare_paddle_dataloader'
]

from typing import Callable, List, Optional, Union, Dict, Sequence

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
if _NEED_IMPORT_PADDLE:
    from paddle.io import DataLoader, Dataset
    from paddle.fluid.dataloader.collate import default_collate_fn
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset
    from fastNLP.core.utils.dummy_class import DummyClass as DataLoader

from fastNLP.core.collators.collator import _MultiCollator
from fastNLP.core.utils.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet


class _PaddleDataset(Dataset):
    """
    对用户传的dataset进行封装，以便Fdataloader能够支持使用自定义的dataset使用paddle的dataloader
    """

    def __init__(self, dataset) -> None:
        super(_PaddleDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        return (item, self.dataset[item])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, item):
        try:
            self.dataset.__getattribute__(item)
        except Exception as e:
            raise e


class PaddleDataLoader(DataLoader):

    def __init__(self, dataset, feed_list=None, places=None,
                 return_list: bool = True, batch_sampler=None,
                 batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False, collate_fn: Callable = None,
                 num_workers: int = 0, use_buffer_reader: bool = True,
                 use_shared_memory: bool = True, timeout: int = 0,
                 worker_init_fn: Callable = None, persistent_workers=False) -> None:

        if not isinstance(dataset, _PaddleDataset):
            dataset = _PaddleDataset(dataset)

        super(PaddleDataLoader, self).__init__(dataset=dataset, feed_list=feed_list, places=places,
                                               return_list=return_list, batch_sampler=batch_sampler,
                                               batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                               collate_fn=None, num_workers=num_workers,
                                               use_buffer_reader=use_buffer_reader, use_shared_memory=use_shared_memory,
                                               timeout=timeout, worker_init_fn=worker_init_fn,
                                               persistent_workers=persistent_workers)
        if isinstance(dataset.dataset, FDataSet):
            self._collate_fn = dataset.dataset.get_collator()
            self._collate_fn.set_as_numpy(as_numpy=True)
            if collate_fn is not None:
                self._collate_fn.add_collator(collate_fn)
        else:
            self._collate_fn = _MultiCollator(collate_fn)
        # _collate_fn = _MultiCollator(AutoCollator(as_numpy=True))
        # if collate_fn is not None:
        #     _collate_fn.add_collator(collate_fn)
        # self._collate_fn = _collate_fn
        self.cur_batch_indices = None

    def __iter__(self):
        # 如果没有auto_collator 也没有自定义collate_fn， 那么此时采用dataloader自带的collate_fn， 将数据打包即可。
        if len(self._collate_fn.get_collators()) == 0:
            self._collate_fn.add_collator(default_collate_fn)
            # self._collate_fn = default_collate_fn
        self.collate_fn = indice_collate_wrapper(self._collate_fn)
        for indices, data in super().__iter__():
            self.cur_batch_indices = indices
            yield data

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

    def set_pad_val(self, *field_names, val: Optional[int] = 0) -> None:
        """
        设置每个field_name的padding值，默认为0，只有当autocollate存在时该方法有效， 若没有则会添加auto_collator函数
        当val=None时，意味着给定的field_names都不需要尝试padding

        :param field_names:
        :param val: padding值，默认为0
        :return:
        """
        for field_name in field_names:
            self._collate_fn.set_pad_val(field_name, val=val)

    def set_input(self, *field_names) -> None:
        """
        被设置为inputs的field_names，会输入到AutoCollator中，未被设置默认过滤掉

        :param field_names:
        :return:
        """
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


def prepare_paddle_dataloader(ds_or_db, feed_list=None, places=None,
                       return_list: bool = True, batch_sampler=None,
                       train_batch_size: int = 1, shuffle: bool = False,
                       drop_last: bool = False, collate_fn: Callable = None,
                       num_workers: int = 0, use_buffer_reader: bool = True,
                       use_shared_memory: bool = True, timeout: int = 0,
                       worker_init_fn: Callable = None, persistent_workers=False,
                       non_train_batch_size: int = 16,
                       input_fields: Union[List[str], str] = None)\
                       -> Union[Sequence[PaddleDataLoader], Dict[str, PaddleDataLoader], PaddleDataLoader]:
    if isinstance(input_fields, str):
        input_fields = [input_fields]

    if isinstance(ds_or_db, Dataset):
        ...
    elif isinstance(ds_or_db, Sequence):
        ds_seq = []
        for ds in ds_or_db:
            dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                  batch_sampler=batch_sampler, batch_size=train_batch_size, shuffle=shuffle,
                                  drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                  use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                  timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
            dl.set_input(*input_fields)
            ds_seq.append(dl)
        return ds_seq

    elif isinstance(ds_or_db, Dict):
        ds_dict = {}
        for name, ds in ds_or_db.items():
            if 'train' in name:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler, batch_size=train_batch_size, shuffle=shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
            else:
                dl = PaddleDataLoader(ds, feed_list=feed_list, places=places, return_list=return_list,
                                      batch_sampler=batch_sampler, batch_size=non_train_batch_size, shuffle=shuffle,
                                      drop_last=drop_last, collate_fn=collate_fn, num_workers=num_workers,
                                      use_shared_memory=use_shared_memory, use_buffer_reader=use_buffer_reader,
                                      timeout=timeout, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
            dl.set_input(*input_fields)
            ds_dict[name] = dl
        return ds_dict
    else:
        raise ValueError(f"ds_or_db: {ds_or_db} must be fastnlp dataset or data_bundle or sequence or mapping!")

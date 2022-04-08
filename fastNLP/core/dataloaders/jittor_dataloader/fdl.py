__all__ = [
    'JittorDataLoader',
    'prepare_jittor_dataloader'
]

from typing import Callable, Optional, List

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    from jittor.dataset.utils import collate_batch
    from jittor.dataset import Dataset
else:
    from fastNLP.core.dataset import DataSet as Dataset
from fastNLP.core.utils.jittor_utils import jittor_collate_wraps
from fastNLP.core.collators import AutoCollator
from fastNLP.core.utils.utils import indice_collate_wrapper
from fastNLP.core.dataset import DataSet as FDataSet


class _JittorDataset(Dataset):
    """
    对用户传的dataset进行封装，以便JittorDataLoader能够支持使用自定义的dataset使用jittor的dataset
    """

    def __init__(self, dataset) -> None:
        super(_JittorDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, item):
        return (item, self.dataset[item])

    def __len__(self) -> int:
        return len(self.dataset)

    # def __getattr__(self, item):
    #     # jittor的Dataset没有的方法而用户的dataset存在且实现了getattribute方法，此时用户可以调用
    #     try:
    #         self.dataset.__getattribute__(item)
    #     except Exception as e:
    #         raise e


class JittorDataLoader:
    """
    提供给使用jittor框架的DataLoader函数，提供了auto_collate的功能， 支持实现了__getitem__和__len__的dataset
    """

    def __init__(self, dataset, batch_size: int = 16, shuffle: bool = False,
                 drop_last: bool = False, num_workers: int = 0, buffer_size: int = 512 * 1024 * 1024,
                 stop_grad: bool = True, keep_numpy_array: bool = False, endless: bool = False,
                 collate_fn: Callable = None) -> None:
        """

        :param dataset: 实现__getitem__和__len__的dataset
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据集
        :param drop_last: 是否去掉最后一个不符合batch_size的数据
        :param num_workers: 进程的数量，当num_workers=0时不开启多进程
        :param buffer_size:
        :param stop_grad:
        :param keep_numpy_array:
        :param endless:
        :param collate_fn: 对取得到的数据进行打包的callable函数
        :param as_numpy: 返回数据是否设置为numpy类型，否则为torch.tensor类型
        """
        # TODO 支持fastnlp dataset
        # TODO 验证支持replacesampler （以后完成）
        # 是否为 jittor 类型的 dataset

        if isinstance(dataset, FDataSet):
            collator = dataset.get_collator().set_as_numpy(as_numpy=True)
        else:
            collator = None

        self.dataset = _JittorDataset(dataset)

        self.dataset.set_attrs(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                               num_workers=num_workers, buffer_size=buffer_size, stop_grad=stop_grad,
                               keep_numpy_array=keep_numpy_array, endless=endless)
        if isinstance(self.dataset.dataset, Dataset):
            self.dataset.dataset.set_attrs(batch_size=1)
        # 用户提供了 collate_fn，则会自动代替 jittor 提供 collate_batch 函数
        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = collate_batch
        self.auto_collator = collator
        self.cur_batch_indices = None

    def __iter__(self):
        # TODO 第一次迭代后不能设置collate_fn，设置是无效的
        if self.cur_batch_indices is None:
            self.dataset.set_attrs(collate_batch=indice_collate_wrapper(jittor_collate_wraps(self.collate_fn,
                                                                                             self.auto_collator)))
        for indices, data in self.dataset.__iter__():
            self.cur_batch_indices = indices
            yield data

    def __len__(self):
        if self.dataset.drop_last:
            return len(self.dataset) // self.dataset.batch_size
        return (len(self.dataset) - 1) // self.dataset.batch_size + 1

    def set_pad_val(self, *field_names, val: Optional[int] = 0) -> None:
        """
        设置每个field_name的padding值，默认为0，只有当autocollate存在时该方法有效， 若没有则会添加auto_collator函数
        当val=None时，意味着给定的field_names都不需要尝试padding

        :param field_names:
        :param val: padding值，默认为0
        :return:
        """
        if self.auto_collator is None:
            self.auto_collator = AutoCollator(as_numpy=True)
        self.auto_collator.set_pad_val(*field_names, val=val)

    def set_input(self, *field_names) -> None:
        """
        被设置为inputs的field_names，会输入到AutoCollator中，未被设置默认过滤掉

        :param field_names:
        :return:
        """
        if self.auto_collator is None:
            self.auto_collator = AutoCollator(as_numpy=True)

        self.auto_collator.set_input(*field_names)

    def get_batch_indices(self) -> List[int]:
        """
        获取当前数据的idx

        :return:
        """
        return self.cur_batch_indices


def prepare_jittor_dataloader():
    ...

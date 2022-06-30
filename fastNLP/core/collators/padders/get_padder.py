from typing import Sequence, Any, Union, Dict
from abc import ABC

from fastNLP.core.log import logger


from .padder import Padder, NullPadder
from .numpy_padder import NumpyNumberPadder, NumpySequencePadder, NumpyTensorPadder
from .torch_padder import TorchNumberPadder, TorchSequencePadder, TorchTensorPadder
from .raw_padder import RawNumberPadder, RawSequencePadder, RawTensorPadder
from .paddle_padder import PaddleTensorPadder, PaddleSequencePadder, PaddleNumberPadder
from .jittor_padder import JittorTensorPadder, JittorSequencePadder, JittorNumberPadder
from .oneflow_padder import OneflowTensorPadder, OneflowSequencePadder, OneflowNumberPadder
from .exceptions import *


def get_padder(batch_field:Sequence[Any], pad_val, dtype, backend, field_name)->Padder:
    """
    根据 参数 与 batch_field ，返回适合于当前 batch_field 的 padder 。

    :param batch_field: 将某 field 的内容组合成一个 batch 传入。
    :param pad_val:
    :param backend:
    :param dtype:
    :param field_name: 方便报错的。
    :return:
    """
    try:
        assert len(batch_field)!=0, "Empty batch encountered."
        logger.debug(f"The content in the field:`{field_name}` is:\n" + str(batch_field))
        if pad_val is None:
            logger.debug(f"The pad_val for field:{field_name} is None, not padding this field.")
            return NullPadder()
        if backend is None:
            logger.debug(f"The backend for field:{field_name} is None, not padding this field.")
            return NullPadder()

        # 首先判断当前 field 是否是必须要 pad ，根据用户设置的 pad_val、dtype 等判断。
        must_pad = False
        if pad_val != 0 or dtype is not None:
            must_pad = True

        catalog = _get_element_shape_dtype(batch_field)  # 首先获取数据的基本信息。

        # 根据 catalog 来判定当前是否可以进行 pad 。
        # 首先检查是否所有的 key 是一样长的，表明深度是一致的
        depths = set(map(len, catalog.keys()))
        num_depth = len(depths)
        if num_depth != 1:
            msg = f'Field:`{field_name}` cannot pad, since it has various depths({depths}) of data. To view more ' \
                  f"information please set logger's level to DEBUG."
            if must_pad:
                raise InconsistencyError(msg)
            raise NoProperPadderError(msg)

        # 再检查所有的元素 shape 是否一致？
        shape_lens = set([len(v[0]) for v in catalog.values()])
        num_shape = len(shape_lens)
        if num_shape != 1:
            msg = f'Field:`{field_name}` cannot pad, since it has various shape length({shape_lens}) of data. To view more ' \
                  f"information please set logger's level to DEBUG."
            if must_pad:
                raise InconsistencyError(msg)
            raise NoProperPadderError(msg)

        # 再检查所有的元素 type 是否一致
        try:
            ele_dtypes = set([v[1] for v in catalog.values()])
        except TypeError:
            ele_dtypes = set([str(v[1]) for v in catalog.values()])
        num_eletypes = len(ele_dtypes)
        if num_eletypes != 1:
            msg = f'Field:`{field_name}` cannot pad, since it has various types({ele_dtypes}) of data. To view more ' \
                  f"information please set logger's level to DEBUG."
            if must_pad:
                raise InconsistencyError(msg)
            raise NoProperPadderError(msg)

        depth = depths.pop()
        shape_len = shape_lens.pop()
        ele_dtype = list(catalog.values())[0][1]  # 因为上面有except的情况，所以这样处理了

        # 需要由 padder 自己决定是否能够 pad 。
        if depth == 1 and shape_len == 0:  # 形如 [0, 1, 2] 或 [True, False, True]
            if backend == 'raw':
                return RawNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'numpy':
                return NumpyNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'torch':
                return TorchNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'paddle':
                return PaddleNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'jittor':
                return JittorNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'oneflow':
                return OneflowNumberPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            else:
                raise ValueError(f"backend={backend} is not supported for list(Field:{field_name}).")

        if depth > 1 and shape_len == 0:  # 形如 [[0, 1], [2]] 这种
            if backend == 'raw':
                return RawSequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'numpy':
                return NumpySequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'torch':
                return TorchSequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'paddle':
                return PaddleSequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'jittor':
                return JittorSequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'oneflow':
                return OneflowSequencePadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            else:
                raise ValueError(f"backend={backend} is not supported for nested list(Field:{field_name}).")

        # 如果有有 shape 的话，只有当该对象拥有 tolist() 方法才行
        if depth == 1 and shape_len != 0 and callable(getattr(batch_field[0], 'tolist', None)):
            if backend == 'raw':
                return RawTensorPadder(pad_val=pad_val, ele_dtype=None, dtype=dtype)
            elif backend == 'numpy':
                return NumpyTensorPadder(pad_val=pad_val, ele_dtype=None, dtype=dtype)
            elif backend == 'torch':
                # 这里 ele_dtype 传入为 None 的原因是防止出现 paddle tensor 转换为 torch tensor
                return TorchTensorPadder(pad_val=pad_val, ele_dtype=None, dtype=dtype)
            elif backend == 'paddle':
                return PaddleTensorPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'jittor':
                return JittorTensorPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            elif backend == 'oneflow':
                return OneflowTensorPadder(pad_val=pad_val, ele_dtype=ele_dtype, dtype=dtype)
            else:
                raise ValueError(f"backend={backend} is not supported for tensors(Field:{field_name}).")

        if shape_len != 0 and depth>1:
            msg = "Does not support pad tensor under nested list. If you need this, please report."
            if must_pad:
                raise RuntimeError(msg)
            raise NoProperPadderError(msg)

    except DtypeError as e:
        msg = f"Fail to get padder for field:{field_name}. " + e.msg + " To view more " \
              "information please set logger's level to DEBUG."
        if must_pad:
            logger.error(msg)
            raise type(e)(msg=msg)

    except NoProperPadderError as e:
        logger.debug(f"{e.msg}")

    except BaseException as e:
        raise e

    return NullPadder()


class HasShapeDtype(ABC):
    """
    检测拥有 shape 和 dtype 属性的对象。一般就是 np.ndarray 或者各类 tensor 。

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is HasShapeDtype:
            if hasattr(subclass, 'shape') and hasattr(subclass, 'dtype'):
                return True
            return False
        return NotImplemented


def _get_element_shape_dtype(content, parent=None, catalog=None)->Dict:
    """
    获取对象的中 element 的基本信息，用于判断是否可以 padding。

    :param content:
    :param tuple parent:
    :param dict catalog: 记录元素信息的 dict。其中的 index 记录的是每一个元素的 拓扑 结构。
        例如: [1, 2, 3] -> {(0,): ((), <class 'int'>), (1,): ((), <class 'int'>), (2,): ((), <class 'int'>)}
        例如: [1, [2, 3], 4] -> {(0,): ((), <class 'int'>), (1, 0): ((), <class 'int'>), (1, 1): ((), <class 'int'>), (2,): ((), <class 'int'>)}
        例如: [[1, 2], [3], [4, 5]] -> {(0, 0): ((), <class 'int'>), (0, 1): ((), <class 'int'>), (1, 0): ((), <class 'int'>), (2, 0): ((), <class 'int'>), (2, 1): ((), <class 'int'>)}
        例如: [torch.ones(3, 4), torch.ones(3, 4), torch.ones(3, 4)]
            -> {(0,): (torch.Size([3, 4]), torch.float32), (1,): (torch.Size([3, 4]), torch.float32), (2,): (torch.Size([3, 4]), torch.float32)}

    :return:
    """
    if catalog is None:
        catalog = {}

    if parent is None:
        parent = ()

    if isinstance(content, HasShapeDtype):  # 各类 tensor 或者 np.ndarray
        shape = content.shape
        dtype = content.dtype
        catalog[parent] = (shape, dtype)
    elif isinstance(content, (tuple, list)):
        for i, c in enumerate(content):
            _get_element_shape_dtype(c, parent=parent + (i,), catalog=catalog)
    else:  # 包括 int/float/bool/dict 以及 其它无法pad 的等
        catalog[parent] = ((), type(content))  # () 表示 shape 的长度为 0，后面表示其类别
    return catalog

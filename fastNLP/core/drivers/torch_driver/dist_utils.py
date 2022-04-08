import io
import pickle
from typing import Mapping
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler
from abc import ABC
from typing import Any, Union, List
import numpy as np
from fastNLP.envs.imports import _TORCH_GREATER_EQUAL_1_8


from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch import distributed as dist

from fastNLP.core.utils import apply_to_collection



def all_gather_object(object_list, obj, group=None):
    """
    Gathers picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the object
    must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        object (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    current_device = torch.device("cpu")
    if dist.is_nccl_available() and isinstance(
            group or dist.distributed_c10d._get_default_group(), dist.ProcessGroupNCCL
    ):
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device("cuda", torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    dist.all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def _validate_output_list_for_rank(my_rank, dst, gather_list):
    if dst == my_rank:
        if not gather_list:
            raise ValueError(
                "Argument ``gather_list`` must be specified on destination rank."
            )
    elif gather_list:
        raise ValueError(
            "Argument ``gather_list`` must NOT be specified "
            "on non-destination ranks."
        )


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    """
    Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank. (default is 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: Note that this API is not supported when using the NCCL backend.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None,
                dst=0
            )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = dist.get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = dist.get_backend(group)
    current_device = torch.device("cpu")
    is_nccl_backend = group_backend == dist.Backend.NCCL
    if is_nccl_backend:
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist.get_world_size(group=group)
    object_sizes_tensor = torch.zeros(group_size, dtype=torch.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=current_device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    dist.gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,
        dst=dst,
        group=group,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)  # type: ignore[call-overload]
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


def _object_to_tensor(obj, device=None):
    f = io.BytesIO()
    _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    if device is not None:
        byte_tensor = byte_tensor.to(device)
        local_size = local_size.to(device)
    return byte_tensor, local_size


def _tensor_to_object(tensor, tensor_size):
    buf = tensor.detach().cpu().numpy().tobytes()[:tensor_size]
    return _unpickler(io.BytesIO(buf)).load()


def send_recv_object(obj, src, cur_rank, device, group=None, tag=0):
    # src rank send to all other ranks
    size = torch.LongTensor([0]).to(device)

    if cur_rank == src:
        world_size = dist.get_world_size(group=group)
        tensor, size = _object_to_tensor(obj)
        tensor = tensor.to(device)
        size = size.to(device)

        # 首先同步 obj 的 size 的信息；
        dist.broadcast(size, src, group=group)
        for subrank in range(world_size):
            if subrank != src:
                dist.send(tensor=tensor, dst=subrank, group=group, tag=tag)
    else:
        dist.broadcast(size, src, group=group)
        tensor = torch.ByteTensor([0] * size).to(device)
        dist.recv(tensor=tensor, src=src, group=group, tag=tag)

    return _tensor_to_object(tensor.cpu(), size)


def _all_gather(obj, **kwargs):
    group = kwargs.get('group', None)
    if isinstance(obj, torch.Tensor):
        gathered_tensor = [torch.zeros_like(obj) for _ in
                           range(torch.distributed.get_world_size(group=group))]

        torch.distributed.all_gather(gathered_tensor, obj, group=group)

        return gathered_tensor

    elif isinstance(obj, tuple) and isinstance(obj[1], torch.Tensor):
        tensor, size = obj
        # 首先需要同步 size 吧？
        group_size = dist.get_world_size(group=group)
        object_sizes_tensor = torch.zeros(
            group_size, dtype=torch.long, device=tensor.device
        )
        object_size_list = [
            object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
        ]
        dist.all_gather(object_size_list, size, group=group)
        max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
        # Resize tensor to max size across all ranks.
        tensor.resize_(max_object_size)
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size, dtype=torch.uint8, device=tensor.device
        )

        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i: max_object_size * (i + 1)]
            for i in range(group_size)
        ]
        dist.all_gather(output_tensors, tensor, group=group)
        object_list = []
        for i, tensor in enumerate(output_tensors):
            tensor = tensor.type(torch.uint8)
            tensor_size = object_size_list[i]
            object_list.append(_tensor_to_object(tensor, tensor_size))
        return object_list
    elif isinstance(obj, tuple) and len(obj) == 2:
        obj, _type = obj
        gathered_tensor = [torch.zeros_like(obj) for _ in
                           range(torch.distributed.get_world_size(group=group))]

        torch.distributed.all_gather(gathered_tensor, obj, group=group)

        if _type == np.ndarray:
            gathered_tensor = [t.detach().cpu().numpy() for t in gathered_tensor]
        else:
            gathered_tensor = [_type(t.item()) for t in gathered_tensor]

        return gathered_tensor
    else:
        raise RuntimeError("Unsupported types to implement all_gather.")


class CanTransferDataType(ABC):
    """
    检测可以进行传输的对象。

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is CanTransferDataType:
            if issubclass(subclass, Mapping):
                return False
            if subclass in (torch.Tensor, tuple, list, str, int, float, bool, np.ndarray):
                return True
            return False
        return NotImplemented


def _tensorize(obj, device=None):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, bool):
        return torch.tensor(obj, dtype=torch.uint8, device=device), bool
    if isinstance(obj, float):
        return torch.tensor(obj, dtype=torch.float, device=device), float
    if isinstance(obj, int):
        return torch.tensor(obj, dtype=torch.int, device=device), int
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj), np.ndarray
    return _object_to_tensor(obj, device)


def _to_device(tensor, device):
    return tensor.contiguous().to(device)


def convert_to_tensors(data: Any, device=None) -> Any:
    data = apply_to_collection(data, CanTransferDataType, _tensorize)
    def _move_to_device_and_make_contiguous(t: Union[torch.Tensor, tuple], device: Union[str, torch.device]):
        if isinstance(t, tuple):
            if isinstance(t[1], torch.Tensor): # 说明是 object 转的
                return t[0].to(device).contiguous(), t[1].to(device)
            else:  # 说明第二个元素是type，见 to_dtype_tensor 函数
                return t[0].to(device).contiguous(), t[1]
        return t.to(device).contiguous()

    data = apply_to_collection(data, (torch.Tensor, tuple), _move_to_device_and_make_contiguous, device=device)
    return data


def fastnlp_torch_all_gather(obj:Any, device=None, group=None)->List:
    """
    实现任何类型的数据都使用该接口可以进行 all_gather 操作。对于非 tensor 类型的数据，通过 pickle 序列化再反序列化的方式进行传输。

    example:
        obj = {
            'a': [1, 1],
            'b': [[1, 2], [1, 2]],
            'c': {
                'd': [1, 2]
            }
        }
        ->
        [
            {'a': 1, 'b':[1, 2], 'c':{'d': 1}},
            {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
        ]

    :param obj: 任意结构的数据，所有的 value 都会变成 list ，其长度为 world_size ，依次为每个 rank 上的对象值
    :param device: 当前 rank 使用的 device 是哪个。为 None 的话默认使用 torch.cuda.current_device() 获取。
    :param group:
    :return: 返回的结果是 [obj0, obj1, ...]，其中 obj_i 即为第 i 个 rank 上的 obj 。
    """
    # # 首先将所有的都移动到cpu上并且连续，防止有 pickle 出问题
    # obj = apply_to_collection(obj, torch.Tensor, _to_device, device=torch.device('cpu'))
    if _TORCH_GREATER_EQUAL_1_8:
        objs = [None for _ in range(dist.get_world_size(group))]
        dist.all_gather_object(objs, obj)
        return objs
    if device is None:
        device = torch.cuda.current_device()
    group = group if group is not None else torch.distributed.group.WORLD
    data = convert_to_tensors(obj, device=device)
    data = apply_to_collection(data, (torch.Tensor, tuple), _all_gather, group=group)

    objs = []

    def _get_obj_on_idx(obj, idx):
        return obj[idx]

    for i in range(dist.get_world_size(group)):
        objs.append(apply_to_collection(data, dtype=list, function=_get_obj_on_idx, idx=i))

    return objs


def fastnlp_torch_broadcast_object(obj, src, device, group=None):
    """
    将 src 上的 obj 对象广播到其它 rank 上。

    :param obj:
    :param src:
    :param device:
    :param group:
    :return:
    """
    cur_rank = dist.get_rank(group)
    # if cur_rank == src:
    #     # 如果有 tensor 全部移动到 cpu 上，方便 pickle
    #     obj = apply_to_collection(obj, torch.Tensor, _to_device, device=torch.device('cpu'))

    if _TORCH_GREATER_EQUAL_1_8:
        if cur_rank!=src:
            get_obj = [None]
            dist.broadcast_object_list(get_obj, src=src, group=group)
            return get_obj[0]
        else:
            dist.broadcast_object_list([obj], src=src, group=group)
            return obj

    if cur_rank == src:
        tensor, size = _object_to_tensor(obj, device=device)
    else:
        size = torch.LongTensor([0]).to(device)

    dist.broadcast(size, src=src, group=group)
    if cur_rank != src:
        tensor = torch.empty(
            size.int().item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=device
        )
    dist.broadcast(tensor, src=src, group=group)

    return _tensor_to_object(tensor, tensor_size=size.item())



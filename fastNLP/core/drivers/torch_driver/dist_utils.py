import io
import pickle
import os
_pickler = pickle.Pickler
_unpickler = pickle.Unpickler
from typing import Any, List

from fastNLP.envs.imports import _TORCH_GREATER_EQUAL_1_8
from fastNLP.core.utils.torch_utils import DEFAULT_TORCH_GROUP
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.envs.env import FASTNLP_NO_SYNC
if _NEED_IMPORT_TORCH:
    import torch
    from torch import distributed as dist
    if _TORCH_GREATER_EQUAL_1_8:
        try:
            from torch._C._distributed_c10d import ProcessGroupGloo
            from torch._C._distributed_c10d import _ProcessGroupWrapper
        except ImportError:
            pass


from fastNLP.core.utils import apply_to_collection


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


def fastnlp_torch_gather_object(obj, dst=0, group=DEFAULT_TORCH_GROUP):
    """
    从其它 rank gather 东西到 dst rank 。

    Example::
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> fastnlp_torch_gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None,
                dst=0
            )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]

    :param obj: 需要发送的 obj 对象，需要是可以 pickable 的对象
    :param dst: 目标的 rank 。
    :param group: 在哪个 group 执行该函数。
    :return: 在 dst 上面返回 world_size 的 list，依次为 rank 0；rank 1...上 obj
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        return [obj]

    if dist.get_rank() == dst:
        object_gather_list = [None for _ in range(dist.get_world_size(group))]
    else:
        object_gather_list = None

    if group is None:
        group = DEFAULT_TORCH_GROUP

    if dist.distributed_c10d._rank_not_in_group(group):
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = dist.get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    # 防止 unpickle 的时候出现在了发送的 gpu 上。
    obj = apply_to_collection(obj, torch.Tensor, _to_device, device=torch.device('cpu'))
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


def _to_device(tensor, device):
    return tensor.contiguous().to(device)


def fastnlp_torch_all_gather(obj: Any, device=None, group=DEFAULT_TORCH_GROUP) ->List:
    """
    实现任何类型的数据都使用该接口可以进行 all_gather 操作。对于非 tensor 类型的数据，通过 pickle 序列化再反序列化的方式进行传输。

    example::

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

    :param obj: 任意结构的数据，如果为 tensor ，需要保证每个显卡上的 tensor 的形状是一样的。如果传入的是非 tensor 对象都将直接进行
        序列化之后进行传输。
    :param device: 当前该参数无意义。
    :param group:
    :return: 返回的结果是 [obj0, obj1, ...]，其中 obj_i 即为第 i 个 rank 上的 obj 。
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        return [obj]

    if group is None:
        group = DEFAULT_TORCH_GROUP
    if isinstance(obj, torch.Tensor):
        objs = [torch.zeros_like(obj) for _ in range(dist.get_world_size(group))]
        dist.all_gather(objs, obj, group=group)
    else:
        objs = [None for _ in range(dist.get_world_size(group))]
        # 防止 unpickle 的时候弄到发送的 gpu 上了
        obj = apply_to_collection(obj, torch.Tensor, _to_device, device=torch.device('cpu'))
        if _TORCH_GREATER_EQUAL_1_8:
            dist.all_gather_object(objs, obj, group=group)
        else:
            objs = all_gather_object(objs, obj, group=group)
    return objs


def fastnlp_torch_broadcast_object(obj, src, device=None, group=DEFAULT_TORCH_GROUP):
    """
    将 src 上的 obj 对象广播到其它 rank 上。

    :param obj: 需要发送的对象
    :param src: 从哪里发出。
    :param device:
    :param group: 属于哪个通信 group
    :return:
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        if src == dist.get_rank(group):
            return obj
        else:
            return None

    if group is None:
        group = DEFAULT_TORCH_GROUP
    cur_rank = dist.get_rank(group)
    if cur_rank == src:
        # 如果有 tensor 全部移动到 cpu 上，方便 pickle , 不然 unpickle 的时候可能会 pickle 到发送过来的卡那里
        obj = apply_to_collection(obj, torch.Tensor, _to_device, device=torch.device('cpu'))
    if _TORCH_GREATER_EQUAL_1_8:
        if cur_rank!=src:
            get_obj = [None]
            dist.broadcast_object_list(get_obj, src=src, group=group)
            return get_obj[0]
        else:
            dist.broadcast_object_list([obj], src=src, group=group)
            return obj
    if device is None:
        device = torch.cuda.current_device()

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


def _check_for_nccl_backend(group):
    pg = group or dist.distributed_c10d._get_default_group()
    # It is not expected for PG to be wrapped many times, but support it just
    # in case
    while isinstance(pg, _ProcessGroupWrapper):
        pg = pg.wrapped_pg

    return (
            dist.is_nccl_available() and
            isinstance(pg, dist.ProcessGroupNCCL)
    )


def all_gather_object(object_list, obj, group=None):
    """
    复制 pytorch 的代码，使得可以版本兼容低版本的 pytorch 。

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]

    :param object_list:
    :param obj:
    :param group:
    :return:
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        return [obj]

    if dist.distributed_c10d._rank_not_in_group(group):
        return
    if _TORCH_GREATER_EQUAL_1_8:
        current_device = torch.device("cpu")
        is_nccl_backend = _check_for_nccl_backend(group)
        if is_nccl_backend:
            # See note about using torch.cuda.current_device() here in docstring.
            # We cannot simply use my_rank since rank == device is not necessarily
            # true.
            current_device = torch.device("cuda", torch.cuda.current_device())
    else:
        current_device = torch.cuda.current_device()

    input_tensor, local_size = _object_to_tensor(obj, device=current_device)

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
    return object_list
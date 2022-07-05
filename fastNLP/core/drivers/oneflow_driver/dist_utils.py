import io
import pickle
import os
from typing import Any, List

from fastNLP.core.utils import apply_to_collection, get_oneflow_device
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from fastNLP.envs.env import FASTNLP_NO_SYNC
if _NEED_IMPORT_ONEFLOW:
    import oneflow
    import oneflow.comm as comm
    import oneflow.env as dist_env

PROTOCOL_VERSION = 1

__all__ = []

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

    obj = {"protocol_version": PROTOCOL_VERSION, "data": obj}
    pickled_bytes = pickle.dumps(obj)

def fastnlp_oneflow_gather_object(obj, dst=0):
    """
    从其它 rank gather 东西到 dst rank 。

    Example::
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> fastnlp_oneflow_gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None,
                dst=0
            )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]

    :param obj: 需要发送的 obj 对象，需要是可以 pickable 的对象
    :param dst: 目标的 rank 。
    :return: 在 dst 上面返回 world_size 的 list，依次为 rank 0；rank 1...上 obj
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        return [obj]

    if dist_env.get_rank() == dst:
        object_gather_list = [None for _ in range(dist_env.get_world_size())]
    else:
        object_gather_list = None

    # Ensure object_gather_list is specified appopriately.
    my_rank = dist_env.get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    # 防止 unpickle 的时候出现在了发送的 gpu 上。
    obj = apply_to_collection(obj, oneflow.Tensor, _to_device, device=oneflow.device("cpu"))
    input_tensor, local_size = _object_to_tensor(obj)
    current_device = oneflow.device("cuda")
    input_tensor = input_tensor.to(current_device)
    local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist_env.get_world_size()
    object_sizes_tensor = oneflow.zeros(group_size, dtype=oneflow.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    comm.all_gather(object_size_list, local_size)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor = input_tensor.reshape(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this rank.
    if my_rank == dst:
        coalesced_output_tensor = oneflow.empty(
            max_object_size * group_size, dtype=oneflow.uint8, device=current_device
        )
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    comm.gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,
        dst=dst,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(oneflow.uint8)  # type: ignore[call-overload]
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


def _object_to_tensor(obj, device=None):
    f = io.BytesIO()
    obj = {"protocol_version": PROTOCOL_VERSION, "data": obj}
    pickled_bytes = pickle.dumps(obj)

    byte_tensor = oneflow.ByteTensor(list(pickled_bytes))
    local_size = oneflow.LongTensor([byte_tensor.numel()])
    if device is not None:
        byte_tensor = byte_tensor.to(device)
        local_size = local_size.to(device)
    return byte_tensor, local_size

def _tensor_to_object(tensor, tensor_size):
    buf = tensor.detach().cpu().numpy().tobytes()[:tensor_size]
    res = pickle.loads(buf)
    assert res["protocol_version"] == PROTOCOL_VERSION
    return res["data"]

def send_recv_object(obj, src, cur_rank, device):
    r"""
    oneflow 中的单点对多点的分发函数；

    例如将进程 0 上的对象 object 分发到其它进程上；

    Example::

        cur_rank = int(os.environ.get('LOCAL_RANK', 0))

        # 拿到 local_device

        send_recv_object(object, 0, cur_rank, local_device)

    :param obj: 一个可以序列化的 python 对象；
    :param src: 从哪一个 rank 上发送到其它 rank；
    :param cur_rank: 当前的进程的 rank 序号；
    :param device: 当前的进程所在的设备；
    :param group: 通信组，默认为 None；
    :param tag: 将发送与远程接收匹配的标记；
    :return:
    """
    # src rank send to all other ranks
    size = oneflow.LongTensor([0]).to(device)

    if cur_rank == src:
        world_size = dist_env.get_world_size()
        tensor, size = _object_to_tensor(obj)
        tensor = tensor.to(device)
        size = size.to(device)

        # 首先同步 obj 的 size 的信息；
        comm.broadcast(size, src)
        for subrank in range(world_size):
            if subrank != src:
                comm.send(tensor=tensor, dst=subrank)
    else:
        comm.broadcast(size, src)
        tensor = oneflow.ByteTensor([0] * size).to(device)
        comm.recv(tensor=tensor, src=src)

    return _tensor_to_object(tensor.cpu(), size)


def _to_device(tensor, device):
    return tensor.contiguous().to(device)


def fastnlp_oneflow_all_gather(obj: Any, device=None) ->List:
    """
    实现任何类型的数据都使用该接口可以进行 all_gather 操作。对于非 tensor 类型的数据，通过 pickle 序列化再反序列化的方式进行传输。

    example::

        >>> # rank 0
        >>> obj = {'a': 1, 'b':[1, 2], 'c':{'d': 1}}
        >>> # rank 1
        >>> obj = {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
        >>> # after all_gather():
        >>> result = [
                {'a': 1, 'b':[1, 2], 'c':{'d': 1}},
                {'a': 1, 'b':[1, 2], 'c':{'d': 2}}
            ]

    :param obj: 任意结构的数据，如果为 tensor ，需要保证每个显卡上的 tensor 的形状是一样的。如果传入的是非 tensor 对象都将直接进行
        序列化之后进行传输。
    :param device: 当前该参数无意义。
    :param group:
    :return: 返回的结果是 [obj0, obj1, ...]，其中 obj_i 即为第 i 个 rank 上的 obj 。
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, "0")) == 2:
        return [obj]

    if isinstance(obj, oneflow.Tensor):
        objs = [oneflow.zeros_like(obj) for _ in range(dist_env.get_world_size())]
        comm.all_gather(objs, obj)
    else:
        objs = [None for _ in range(dist_env.get_world_size())]
        # 防止 unpickle 的时候弄到发送的 gpu 上了
        obj = apply_to_collection(obj, oneflow.Tensor, _to_device, device=oneflow.device("cpu"))
        all_gather_object(objs, obj)
    return objs


def fastnlp_oneflow_broadcast_object(obj, src, device=None):
    """
    将 src 上的 obj 对象广播到其它 rank 上。

    :param obj: 需要发送的对象
    :param src: 从哪里发出。
    :param device:
    :param group: 属于哪个通信 group
    :return:
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, "0")) == 2:
        if src == dist_env.get_rank():
            return obj
        else:
            return None

    cur_rank = dist_env.get_rank()
    if cur_rank == src:
        # 如果有 tensor 全部移动到 cpu 上，方便 pickle , 不然 unpickle 的时候可能会 pickle 到发送过来的卡那里
        obj = apply_to_collection(obj, oneflow.Tensor, _to_device, device=oneflow.device("cpu"))
    if device is None:
        device = oneflow.cuda.current_device()
    device = get_oneflow_device(device)

    if cur_rank == src:
        tensor, size = _object_to_tensor(obj, device=device)
    else:
        size = oneflow.LongTensor([0]).to(device)

    comm.broadcast(size, src=src)
    if cur_rank != src:
        tensor = oneflow.empty(
            size.int().item(),  # type: ignore[arg-type]
            dtype=oneflow.uint8,
            device=device
        )
    comm.broadcast(tensor, src=src)

    return _tensor_to_object(tensor, tensor_size=size.item())

def all_gather_object(object_list, obj):
    """

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
    if int(os.environ.get(FASTNLP_NO_SYNC, "0")) == 2:
        return [obj]
    
    current_device = get_oneflow_device(oneflow.cuda.current_device())

    input_tensor, local_size = _object_to_tensor(obj, device=current_device)

    # Gather all local sizes. This is so that we can find the max size, and index
    # until the correct size when deserializing the tensors.
    group_size = dist_env.get_world_size()
    object_sizes_tensor = oneflow.zeros(
        group_size, dtype=oneflow.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    comm.all_gather(object_size_list, local_size)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor = input_tensor.reshape(max_object_size)
    coalesced_output_tensor = oneflow.empty(
        max_object_size * group_size, dtype=oneflow.uint8, device=current_device
    )
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    comm.all_gather(output_tensors, input_tensor)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(oneflow.uint8)
        if tensor.device != oneflow.device("cpu"):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)
    return object_list

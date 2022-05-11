import io
import pickle
import os
from typing import Any, List

import numpy as np
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.env import FASTNLP_NO_SYNC
from fastNLP.core.utils import paddle_move_data_to_device

if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.distributed as dist
    from paddle.framework.io import (
        _is_state_dict,
        _build_saved_state_dict,
        _unpack_saved_dict,
        _pickle_save,
        _pack_loaded_dict,
        _ndarray_to_tensor,
        _parse_load_result,
    )

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

def paddle_pickle_dump(obj, stream, protocol):
    """
    Reference to `paddle.save`
    """
    if _is_state_dict(obj):
        saved_obj = _build_saved_state_dict(obj)
        saved_obj = _unpack_saved_dict(saved_obj, protocol)
        pickle.dump(saved_obj, stream, protocol=protocol)
    else:
        _pickle_save(obj, stream, protocol)

def paddle_pickle_load(stream):
    """
    Reference to `paddle.load`
    """
    load_result = pickle.load(stream)
    if isinstance(load_result, dict):
        load_result = _pack_loaded_dict(load_result)
        if "StructuredToParameterName@@" in load_result:

            for key in load_result["StructuredToParameterName@@"]:
                if isinstance(load_result[key], np.ndarray):
                    load_result[key] = _ndarray_to_tensor(
                        load_result[key], return_numpy=False)

            if "StructuredToParameterName@@" in load_result:
                del load_result["StructuredToParameterName@@"]
        else:
            load_result = _parse_load_result(load_result, return_numpy=False)

    else:
        load_result = _parse_load_result(load_result, return_numpy=False)

    return load_result

def _object_to_tensor(obj, device=None):
    f = io.BytesIO()
    paddle_pickle_dump(obj, f, protocol=2)
    byte_data = list(f.getvalue())
    byte_tensor = paddle.to_tensor(byte_data, dtype=paddle.int32)
    local_size = paddle.to_tensor([byte_tensor.numel()])
    if device is not None:
        byte_tensor = paddle_move_data_to_device(byte_tensor, device)
        local_size = paddle_move_data_to_device(local_size, device)
    return byte_tensor, local_size

def _tensor_to_object(tensor, tensor_size):
    buf = tensor.astype(paddle.uint8).detach().cpu().numpy().tobytes()[:tensor_size]
    return paddle_pickle_load(io.BytesIO(buf))

def fastnlp_paddle_gather_object(obj, dst=0, group=None):
    """
    从其它 rank gather 东西到 dst rank 。

    Example::
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> fastnlp_paddle_gather_object(
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
        object_gather_list = [None for _ in range(dist.get_world_size())]
    else:
        object_gather_list = None

    # if group is None:
        # TODO 2.2 版本存在 bug
        # group = dist.collective._get_global_group()

    if group is not None and not group.is_member():
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = dist.get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    # 防止 unpickle 的时候出现在了发送的 gpu 上。
    obj = paddle_move_data_to_device(obj, device="cpu")
    input_tensor, local_size = _object_to_tensor(obj)
    # 目前 paddle 的 group 仅支持 nccl
    input_tensor = paddle_move_data_to_device(input_tensor, device=paddle.device.get_device())
    local_size = paddle_move_data_to_device(local_size, device=paddle.device.get_device())

    # 收集所有的 local_size，找到最大的 size
    object_size_list = []
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    input_tensor.reshape_(max_object_size)
    # TODO 暂时没有在 paddle 中发现类似 torch.distributed.gather 的函数
    output_tensors = []
    dist.all_gather(output_tensors, input_tensor, group)
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.astype(paddle.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)

def send_recv_object(obj, src, cur_rank, device, group=None, use_calc_stream=True):
    # src rank send to all other ranks
    size = paddle_move_data_to_device(paddle.to_tensor([0]), device)

    if cur_rank == src:
        world_size = dist.get_world_size()
        tensor, size = _object_to_tensor(obj)
        tensor = tensor.to(device)
        size = size.to(device)

        # 首先同步 obj 的 size 的信息；
        dist.broadcast(size, src, group=group)
        for subrank in range(world_size):
            if subrank != src:
                dist.send(tensor=tensor, dst=subrank, group=group, use_calc_stream=use_calc_stream)
    else:
        dist.broadcast(size, src, group=group)
        tensor = paddle_move_data_to_device(paddle.to_tensor([0] * size), device)
        dist.recv(tensor=tensor, src=src, group=group, use_calc_stream=use_calc_stream)

    return _tensor_to_object(tensor.cpu(), size)

def fastnlp_paddle_all_gather(obj: Any, device=None, group=None) ->List:
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

    # if group is None:
        # TODO 2.2 版本存在 bug
        # group = dist.collective._get_global_group()
    if isinstance(obj, paddle.Tensor):
        objs = []
        dist.all_gather(objs, obj, group=group)
    else:
        objs = [None for _ in range(dist.get_world_size())]
        # 防止 unpickle 的时候弄到发送的 gpu 上了
        obj = paddle_move_data_to_device(obj, "cpu")
        objs = all_gather_object(objs, obj, group=group)

    return objs


def fastnlp_paddle_broadcast_object(obj, src, device=None, group=None):
    """
    将 src 上的 obj 对象广播到其它 rank 上。

    :param obj: 需要发送的对象
    :param src: 从哪里发出。
    :param device:
    :param group: 属于哪个通信 group
    :return:
    """
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        if src == dist.get_rank():
            return obj
        else:
            return None

    cur_rank = dist.get_rank()
    if cur_rank == src:
        # 如果有 tensor 全部移动到 cpu 上，方便 pickle , 不然 unpickle 的时候可能会 pickle 到发送过来的卡那里
        obj = paddle_move_data_to_device(obj, "cpu")

    if device is None:
        device = paddle.device.get_device()

    if cur_rank == src:
        tensor, size = _object_to_tensor(obj, device=device)
    else:
        size = paddle_move_data_to_device(paddle.to_tensor([0]), device)

    dist.broadcast(size, src=src, group=group)
    if cur_rank != src:
        tensor = paddle.empty(
            size.astype(paddle.int32),  # type: ignore[arg-type]
            dtype=paddle.int32,
        )
    dist.broadcast(tensor, src=src, group=group)

    return _tensor_to_object(tensor, tensor_size=size.item())

def all_gather_object(object_list, obj, group=None):
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
    if int(os.environ.get(FASTNLP_NO_SYNC, '0')) == 2:
        return [obj]

    if group is not None and not group.is_member():
        return
    
    current_device = paddle.device.get_device()

    input_tensor, local_size = _object_to_tensor(obj, device=current_device)

    # 聚合 tensor 的 size，找到最大的
    object_size_list = []
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # 将张量进行 pad
    pad_dims = []
    pad_by = (max_object_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    tensor_padded = paddle.nn.functional.pad(input_tensor, pad_dims)

    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = []
    dist.all_gather(output_tensors, tensor_padded, group=group)
    dist.barrier()
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.astype(paddle.uint8)
        if not tensor.place.is_cpu_place():
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)
    return object_list
import warnings
from typing import Any, Optional, Union

import numpy as np

from fastNLP.core.utils import paddle_to, apply_to_collection
from fastNLP.core.log import logger
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_TORCH, _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_JITTOR:
    import jittor

if _NEED_IMPORT_TORCH:
    import torch

__all__ = [
    "paddle2torch",
    "torch2paddle",
    "jittor2torch",
    "torch2jittor",
]

def _paddle2torch(paddle_tensor: 'paddle.Tensor', device: Optional[Union[str, int]] = None, no_gradient: bool = None) -> 'torch.Tensor':
    """
    将 :class:`paddle.Tensor` 转换为 :class:`torch.Tensor` ，并且能够保留梯度进行反向传播

    :param paddle_tensor: 要转换的 **paddle** 张量；
    :param device: 是否将转换后的张量迁移到特定设备上，为 ``None``时，和输入的张量相同；
    :param no_gradient: 是否保留原张量的梯度。为 ``None`` 时，新的张量与输入张量保持一致；
        为 ``True`` 时，全部不保留梯度；为 ``False`` 时，全部保留梯度；
    :return: 转换后的 **torch** 张量；
    """
    no_gradient = paddle_tensor.stop_gradient if no_gradient is None else no_gradient
    paddle_numpy = paddle_tensor.numpy()
    if not np.issubdtype(paddle_numpy.dtype, np.inexact):
        no_gradient = True

    if device is None:
        if paddle_tensor.place.is_gpu_place():
            # paddlepaddle有两种Place，对应不同的device id获取方式
            if hasattr(paddle_tensor.place, "gpu_device_id"):
                # paddle.fluid.core_avx.Place
                # 在gpu环境下创建张量的话，张量的place是这一类型
                device = f"cuda:{paddle_tensor.place.gpu_device_id()}"
            else:
                # paddle.CUDAPlace
                device = f"cuda:{paddle_tensor.place.get_device_id()}"
        else:
            # TODO: 可能需要支持xpu等设备
            device = "cpu"

    if not no_gradient:
        # 保持梯度，并保持反向传播
        # torch.tensor会保留numpy数组的类型
        torch_tensor = torch.tensor(paddle_numpy, requires_grad=True, device=device)
        hook = torch_tensor.register_hook(
            lambda grad: paddle.autograd.backward(paddle_tensor, paddle.to_tensor(grad.cpu().numpy()))
        )
    else:
        # 不保留梯度
        torch_tensor = torch.tensor(paddle_numpy, requires_grad=False, device=device)

    return torch_tensor


def _torch2paddle(torch_tensor: 'torch.Tensor', device: str = None, no_gradient: bool = None) -> 'paddle.Tensor':
    """
    将 :class:`torch.Tensor` 转换为 :class:`paddle.Tensor`，并且能够保留梯度进行反向传播。

    :param torch_tensor: 要转换的 **torch** 张量；
    :param device: 是否将转换后的张量迁移到特定设备上，输入为 ``None`` 时，和输入的张量相同；
    :param no_gradient: 是否保留原张量的梯度。为 ``None`` 时，新的张量与输入张量保持一致；
        为 ``True`` 时，全部不保留梯度；为 ``False`` 时，全部保留梯度；
    :return: 转换后的 **paddle** 张量；
    """
    no_gradient = not torch_tensor.requires_grad if no_gradient is None else no_gradient
    if device is None:
        if torch_tensor.is_cuda:
            device = f"gpu:{torch_tensor.device.index}"
        else:
            device = "cpu"

    if not no_gradient:
        # 保持梯度并保持反向传播
        # paddle的stop_gradient和torch的requires_grad表现是相反的
        paddle_tensor = paddle.to_tensor(torch_tensor.detach().cpu().numpy(), stop_gradient=False)
        hook = paddle_tensor.register_hook(
            lambda grad: torch.autograd.backward(torch_tensor, torch.tensor(grad.numpy()))
        )
    else:
        paddle_tensor = paddle.to_tensor(torch_tensor.detach().cpu().numpy(), stop_gradient=True)

    paddle_tensor = paddle_to(paddle_tensor, device)

    return paddle_tensor


def _jittor2torch(jittor_var: 'jittor.Var', device: Optional[Union[str, int]] = None, no_gradient: bool = None) -> 'torch.Tensor':
    """
    将 :class:`jittor.Var` 转换为 :class:`torch.Tensor` 。

    :param jittor_var: 要转换的 **jittor** 变量；
    :param device: 是否将转换后的张量迁移到特定设备上，输入为 ``None`` 时，根据 ``jittor.flags.use_cuda`` 决定；
    :param no_gradient: 是否保留原张量的梯度。为``None``时，新的张量与输入张量保持一致；
        为 ``True`` 时，全部不保留梯度；为 ``False`` 时，全部保留梯度；
    :return: 转换后的 **torch** 张量；
    """
    # TODO: warning：无法保留梯度
    # jittor的grad可以通过callback进行传递
    # 如果outputs有_grad键，可以实现求导
    no_gradient = not jittor_var.requires_grad if no_gradient is None else no_gradient
    if no_gradient == False:
        warnings.warn("The result tensor will not keep gradients due to differences between jittor and pytorch.")
    jittor_numpy = jittor_var.numpy()
    if not np.issubdtype(jittor_numpy.dtype, np.inexact):
        no_gradient = True

    if device is None:
        # jittor的设备分配是自动的
        # 根据use_cuda判断
        if jittor.flags.use_cuda:
            device = "cuda:0"
        else:
            device = "cpu"

    torch_tensor = torch.tensor(jittor_numpy, requires_grad=not no_gradient, device=device)

    return torch_tensor


def _torch2jittor(torch_tensor: 'torch.Tensor', no_gradient: bool = None) -> 'jittor.Var':
    """
    将 :class:`torch.Tensor` 转换为 :class:`jittor.Var` 。

    :param torch_tensor: 要转换的 **torch** 张量；
    :param no_gradient: 是否保留原张量的梯度。为``None``时，新的张量与输入张量保持一致；
        为 ``True`` 时，全部不保留梯度；为 ``False`` 时，全部保留梯度；
    :return: 转换后的 **jittor** 变量；
    """
    no_gradient = not torch_tensor.requires_grad if no_gradient is None else no_gradient

    if not no_gradient:
        # 保持梯度并保持反向传播
        jittor_var = jittor.Var(torch_tensor.detach().numpy())
        jittor_var.requires_grad = True
        hook = jittor_var.register_hook(
            lambda grad: torch.autograd.backward(torch_tensor, torch.tensor(grad.numpy()))
        )
    else:
        jittor_var = jittor.Var(torch_tensor.detach().numpy())
        jittor_var.requires_grad = False

    return jittor_var


def torch2paddle(batch: Any, device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的 :class:`torch.Tensor` 转换为 :class:`paddle.Tensor` 。

    :param batch: 包含 :class:`torch.Tensor` 类型的数据集合
    :param device: 是否将转换后的张量迁移到特定设备上。为 ``None`` 时，和输入保持一致；
    :param no_gradient: 是否保留原张量的梯度。为 ``None`` 时，新的张量与输入张量保持一致；
        为 ``True`` 时，不保留梯度；为 ``False`` 时，保留梯度；
    :return: 转换后的数据；      
    """

    return apply_to_collection(
        batch,
        dtype=torch.Tensor,
        function=_torch2paddle,
        device=device,
        no_gradient=no_gradient,
    )


def paddle2torch(batch: Any, device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的 :class:`paddle.Tensor` 转换为 :class:`torch.Tensor` 。

    :param batch: 包含 :class:`paddle.Tensor` 类型的数据集合；
    :param device: 是否将转换后的张量迁移到特定设备上。为 ``None``时，和输入保持一致；
    :param no_gradient: 是否保留原张量的梯度。为 ``None`` 时，新的张量与输入张量保持一致；
        为 ``True`` 时，不保留梯度；为 ``False`` 时，保留梯度；
    :return: 转换后的数据；    
    """

    return apply_to_collection(
        batch,
        dtype=paddle.Tensor,
        function=_paddle2torch,
        device=device,
        no_gradient=no_gradient,
    )


def jittor2torch(batch: Any, device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的 :class:`jittor.Var` 转换为 :class:`torch.Tensor` 。

    .. note::

        注意，由于 **pytorch** 和 **jittor** 之间的差异，从 :class:`jittor.Var` 转换至
        :class:`torch.Tensor` 的过程中无法保留原张量的梯度。

    :param batch: 包含 :class:`jittor.Var` 类型的数据集合；
    :param device: 是否将转换后的张量迁移到特定设备上。为 ``None``时，和输入保持一致；
    :param no_gradient: 是否保留原张量的梯度，在这个函数中该参数无效;
    :return: 转换后的数据；
    """

    return apply_to_collection(
        batch,
        dtype=jittor.Var,
        function=_jittor2torch,
        device=device,
        no_gradient=no_gradient,
    )


def torch2jittor(batch: Any, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的 :class:`torch.Tensor` 转换为 :class:`jittor.Var` 。

    .. note::

        **jittor** 会自动为创建的变量分配设备。

    :param batch: 包含 :class:`torch.Tensor` 类型的数据集合；
    :param no_gradient: 是否保留原张量的梯度。为 ``None`` 时，新的张量与输入张量保持一致；
        为 ``True`` 时，不保留梯度；为 ``False`` 时，保留梯度；
    :return: 转换后的数据； 
    """
    
    return apply_to_collection(
        batch,
        dtype=torch.Tensor,
        function=_torch2jittor,
        no_gradient=no_gradient,
    )
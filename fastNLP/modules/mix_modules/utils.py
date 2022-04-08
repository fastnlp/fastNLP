import warnings
import os
from typing import Any, Optional, Union

import numpy as np

from fastNLP.core.utils.utils import apply_to_collection
from fastNLP.core.utils.paddle_utils import paddle_to
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

def _paddle2torch(paddle_tensor: 'paddle.Tensor', target_device: Optional[Union[str, int]] = None, no_gradient: bool = None) -> 'torch.Tensor':
    """
    将paddle tensor转换为torch tensor，并且能够保留梯度进行反向传播
    :param paddle_tensor: 要转换的paddle张量
    :param target_device: 是否将转换后的张量迁移到特定设备上，输入为`None`时，和输入的张量相同。
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的torch张量
    """
    no_gradient = paddle_tensor.stop_gradient if no_gradient is None else no_gradient
    paddle_numpy = paddle_tensor.numpy()
    if not np.issubdtype(paddle_numpy.dtype, np.inexact):
        no_gradient = True

    if target_device is None:
        if paddle_tensor.place.is_gpu_place():
            # paddlepaddle有两种Place，对应不同的device id获取方式
            if hasattr(paddle_tensor.place, "gpu_device_id"):
                # paddle.fluid.core_avx.Place
                # 在gpu环境下创建张量的话，张量的place是这一类型
                target_device = f"cuda:{paddle_tensor.place.gpu_device_id()}"
            else:
                # paddle.CUDAPlace
                target_device = f"cuda:{paddle_tensor.place.get_device_id()}"
        else:
            # TODO: 可能需要支持xpu等设备
            target_device = "cpu"

    if not no_gradient:
        # 保持梯度，并保持反向传播
        # torch.tensor会保留numpy数组的类型
        torch_tensor = torch.tensor(paddle_numpy, requires_grad=True, device=target_device)
        hook = torch_tensor.register_hook(
            lambda grad: paddle.autograd.backward(paddle_tensor, paddle.to_tensor(grad.cpu().numpy()))
        )
    else:
        # 不保留梯度
        torch_tensor = torch.tensor(paddle_numpy, requires_grad=False, device=target_device)

    return torch_tensor


def _torch2paddle(torch_tensor: 'torch.Tensor', target_device: str = None, no_gradient: bool = None) -> 'paddle.Tensor':
    """
    将torch tensor转换为paddle tensor，并且能够保留梯度进行反向传播。
    :param torch_tensor: 要转换的torch张量
    :param target_device: 是否将转换后的张量迁移到特定设备上，输入为`None`时，和输入的张量相同。
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的paddle张量
    """
    no_gradient = not torch_tensor.requires_grad if no_gradient is None else no_gradient
    if target_device is None:
        if torch_tensor.is_cuda:
            target_device = f"gpu:{torch_tensor.device.index}"
        else:
            target_device = "cpu"

    if not no_gradient:
        # 保持梯度并保持反向传播
        # paddle的stop_gradient和torch的requires_grad表现是相反的
        paddle_tensor = paddle.to_tensor(torch_tensor.detach().numpy(), stop_gradient=False)
        hook = paddle_tensor.register_hook(
            lambda grad: torch.autograd.backward(torch_tensor, torch.tensor(grad.numpy()))
        )
    else:
        paddle_tensor = paddle.to_tensor(torch_tensor.detach().numpy(), stop_gradient=True)

    paddle_tensor = paddle_to(paddle_tensor, target_device)

    return paddle_tensor


def _jittor2torch(jittor_var: 'jittor.Var', target_device: Optional[Union[str, int]] = None, no_gradient: bool = None) -> 'torch.Tensor':
    """
    将jittor Var转换为torch tensor，并且能够保留梯度进行反向传播
    :param jittor_var: 要转换的jittor变量
    :param target_device: 是否将转换后的张量迁移到特定设备上，输入为`None`时，根据jittor.flags.use_cuda决定。
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的torch张量
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

    if target_device is None:
        # jittor的设备分配是自动的
        # 根据use_cuda判断
        if jittor.flags.use_cuda:
            target_device = "cuda:0"
        else:
            target_device = "cpu"

    torch_tensor = torch.tensor(jittor_numpy, requires_grad=not no_gradient, device=target_device)

    return torch_tensor


def _torch2jittor(torch_tensor: 'torch.Tensor', no_gradient: bool = None) -> 'jittor.Var':
    """
    将torch tensor转换为jittor Var，并且能够保留梯度进行反向传播
    :param torch_tensor: 要转换的torch张量
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的jittor变量
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


def torch2paddle(torch_in: Any, target_device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的torch张量转换为paddle张量
    :param torch_in: 要转换的包含torch.Tensor类型的变量
    :param target_device: 是否将转换后的张量迁移到特定设备上，
                          输入为`None`时，和输入的张量相同，
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 将所有torch.Tensor转换为paddle.Tensor的张量           
    """

    return apply_to_collection(
        torch_in,
        dtype=torch.Tensor,
        function=_torch2paddle,
        target_device=target_device,
        no_gradient=no_gradient,
    )


def paddle2torch(paddle_in: Any, target_device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的paddle张量转换为torch张量
    :param torch_in: 要转换的包含paddle.Tensor类型的变量
    :param target_device: 是否将转换后的张量迁移到特定设备上，
                          输入为`None`时，和输入的张量相同，
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 将所有paddle.Tensor转换为torch.Tensor后的变量          
    """

    return apply_to_collection(
        paddle_in,
        dtype=paddle.Tensor,
        function=_paddle2torch,
        target_device=target_device,
        no_gradient=no_gradient,
    )


def jittor2torch(jittor_in: Any, target_device: str = None, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的jittor变量转换为torch张量
    :param jittor_in: 要转换的jittor变量
    :param target_device: 是否将转换后的张量迁移到特定设备上，输入为`None`时，默认为cuda:0。
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的torch张量
    """

    return apply_to_collection(
        jittor_in,
        dtype=jittor.Var,
        function=_jittor2torch,
        target_device=target_device,
        no_gradient=no_gradient,
    )


def torch2jittor(torch_in: Any, no_gradient: bool = None) -> Any:
    """
    递归地将输入中包含的torch张量转换为jittor变量
    :param torch_tensor: 要转换的torch张量
    :param no_gradient: 是否保留原张量的梯度。为`None`时，新的张量与输入张量保持一致；
                        为`True`时，全部不保留梯度；为`False`时，全部保留梯度。
    :return: 转换后的jittor变量
    """
    
    return apply_to_collection(
        torch_in,
        dtype=torch.Tensor,
        function=_torch2jittor,
        no_gradient=no_gradient,
    )
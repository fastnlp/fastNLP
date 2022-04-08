import os
import io
import pickle
from typing import Dict
from collections import OrderedDict

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH
from fastNLP.core.utils.paddle_utils import paddle_to

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.nn import Layer as PaddleLayer

if _NEED_IMPORT_TORCH:
    import torch
    from torch.nn import Module as TorchModule, Parameter as TorchParameter

if _NEED_IMPORT_JITTOR:
    import jittor


__all__ = [
    "MixModule",
]

class MixModule:
    """
    TODO: 支持不同的混合方式；添加state_dict的支持；如果参数里有List of Tensors该怎么处理；
        是否需要仿照Module那样在初始化的时候给各种模型分类
    可以同时使用Torch和Paddle框架的混合模型
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix='', recurse: bool=True, backend=None):
        """
        返回模型的名字和参数
        :param prefix: 输出时在参数名前加上的前缀
        :param recurse: 是否递归地输出参数
        :param backend: `backend`=`None`时，将所有模型和张量的参数返回；
                        `backend`=`torch`时，返回`torch`的参数；
                        `backend`=`paddle`时，返回`paddle`的参数。
        """
        if backend is None:
            generator = self.attributes(TorchModule, TorchParameter, PaddleLayer)
        elif backend == "torch":
            generator = self.attributes(TorchModule, TorchParameter)
        elif backend == "paddle":
            generator = self.attributes(PaddleLayer)
        else:
            raise ValueError("Unknown backend parameter.")

        for name, value in generator:
            name = prefix + ('.' if prefix else '') + name
            if isinstance(value, TorchParameter):
                # 非Module/Layer类型，直接输出名字和值
                yield name, value
            elif recurse:
                # 递归地调用named_parameters
                for name_r, value_r in value.named_parameters(name, recurse):
                    yield name_r, value_r

    def parameters(self, recurse: bool = True, backend: str = None):
        """
        返回模型的参数
        :param recurse:
        :param backend: `backend`=`None`时，将所有模型和张量的参数返回；
                        `backend`=`torch`时，返回`torch`的参数；
                        `backend`=`paddle`时，返回`paddle`的参数。
        """
        for name, value in self.named_parameters(recurse=recurse, backend=backend):
            yield value
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def test_step(self, batch):
        raise NotImplementedError

    def validate_step(self, batch):
        raise NotImplementedError

    def train(self):
        for name, value in self.attributes(TorchModule, PaddleLayer):
            value.train()

    def eval(self):
        for name, value in self.attributes(TorchModule, PaddleLayer):
            value.eval()

    def to(self, device):
        """
        :param device: 设备名
        """
        # 有jittor的话 warning
        if device == "cpu":
            paddle_device = device
        elif device.startswith("cuda"):
            paddle_device = device.replace("cuda", "gpu")
        elif device.startswith("gpu"):
            paddle_device = device
            device = device.replace("gpu", "cuda")
        else:
            raise ValueError("Device value error")

        for name, value in self.attributes(TorchModule):
            # torch的to函数不影响Tensor
            vars(self)[name] = value.to(device)
        for name, value in self.attributes(TorchParameter):
            # Parameter在经过to函数后会变成Tensor类型
            vars(self)[name] = TorchParameter(value.to(device), requires_grad=value.requires_grad)

        for name, value in self.attributes(PaddleLayer):
            vars(self)[name] = value.to(paddle_device)
        for name, value in self.attributes(paddle.Tensor):
        # paddle的to函数会影响到Tensor
            vars(self)[name] = paddle_to(value, paddle_device)

        return self

    def state_dict(self, backend: str = None) -> Dict:
        """
        返回模型的state_dict。
        NOTE: torch的destination参数会在将来删除，因此不提供destination参数
        :param backend: `backend`=`None`时，将所有模型和张量的state dict返回；
                        `backend`=`torch`时，返回`torch`的state dict；
                        `backend`=`paddle`时，返回`paddle`的state dict。
        """
        if backend is None:
            generator = self.attributes(TorchModule, TorchParameter, PaddleLayer)
        elif backend == "torch":
            generator = self.attributes(TorchModule, TorchParameter)
        elif backend == "paddle":
            generator = self.attributes(PaddleLayer)
        else:
            raise ValueError(f"Unknown backend {backend}.")

        destination = OrderedDict()

        for name, value in generator:
            if value is None:
                continue
            if isinstance(value, TorchParameter):
                destination[name] = value
            else:
                # 不同框架state_dict函数的参数名和顺序不同
                if isinstance(value, PaddleLayer):
                    kwargs = {
                        "structured_name_prefix": name + ".",
                    }
                elif isinstance(value, TorchModule):
                    kwargs = {
                        "prefix": name + ".",
                    }
                else:
                    raise ValueError(f"Unknown item type {type(value)}")
                destination.update(value.state_dict(**kwargs))

        return destination

    def save_state_dict_to_file(self, path: str):
        """
        保存模型的state dict到path
        """
        # TODO 设备限制
        filename = os.path.basename(path)
        if filename == "":
            raise ValueError("Received empty filename.")
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        protocol = 4

        saved = {}
        paddle_dict = self.state_dict(backend="paddle")
        torch_dict = self.state_dict(backend="torch")
        # 保存paddle部分
        # 调用paddle保存时的处理函数
        paddle_saved_obj = paddle.framework.io._build_saved_state_dict(paddle_dict)
        paddle_saved_obj = paddle.fluid.io._unpack_saved_dict(paddle_saved_obj, protocol)
        # 将返回的dict保存
        saved["paddle"] = paddle_saved_obj

        # 保存torch部分
        buffer = io.BytesIO()
        torch.save(torch_dict, buffer)
        saved["torch"] = buffer.getvalue()

        # 保存
        with open(path, "wb") as f:
            pickle.dump(saved, f, protocol)

    def load_state_dict_from_file(self, path: str):
        """
        从 `path` 中加载保存的state dict
        """
        state_dict = {}
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        # 加载paddle的数据
        paddle_loaded_obj = loaded["paddle"]
        paddle_load_result = paddle.fluid.io._pack_loaded_dict(paddle_loaded_obj)
        if "StructuredToParameterName@@" in paddle_load_result:
            for key in paddle_load_result["StructuredToParameterName@@"]:
                if isinstance(paddle_load_result[key], np.ndarray):
                    paddle_load_result[key] = paddle.to_tensor(paddle_load_result[key])
        state_dict.update(paddle_load_result)
        # 加载torch的数据
        torch_loaded_obj = loaded["torch"]
        torch_bytes = io.BytesIO(torch_loaded_obj)
        torch_load_result = torch.load(torch_bytes)
        state_dict.update(torch_load_result)

        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """
        从state dict中加载数据
        """
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        new_state = {}

        local_state = self.state_dict()

        # 对字典内容按前缀进行归类
        for key, value in state_dict.items():
            splited = key.split(".", 1)
            if len(splited) == 1:
                # 没有前缀，实际上只有torch.nn.Parameter会进入这种情况
                new_state[key] = value
            else:
                prefix, name = splited
                if prefix not in new_state:
                    new_state[prefix] = {}
                new_state[prefix][name] = value

        for key, param in self.attributes(TorchModule, TorchParameter, PaddleLayer):
            if key in new_state:
                # 在传入的字典中找到了对应的值
                input_param = new_state[key]
                if not isinstance(input_param, dict):
                    # 且不是字典，即上述没有前缀的情况
                    # 按照torch.nn.Module._load_from_state_dict进行赋值
                    if not torch.overrides.is_tensor_like(input_param):
                        error_msgs.append('While copying the parameter named "{}", '
                                        'expected torch.Tensor or Tensor-like object from checkpoint but '
                                        'received {}'
                                        .format(key, type(input_param)))
                        continue

                    # This is used to avoid copying uninitialized parameters into
                    # non-lazy modules, since they dont have the hook to do the checks
                    # in such case, it will error when accessing the .shape attribute.
                    is_param_lazy = torch.nn.parameter.is_lazy(param)
                    # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                    if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                        input_param = input_param[0]

                    if not is_param_lazy and input_param.shape != param.shape:
                        # local shape should match the one in checkpoint
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                        'the shape in current model is {}.'
                                        .format(key, input_param.shape, param.shape))
                        continue
                    try:
                        with torch.no_grad():
                            param.copy_(input_param)
                    except Exception as ex:
                        error_msgs.append('While copying the parameter named "{}", '
                                        'whose dimensions in the model are {} and '
                                        'whose dimensions in the checkpoint are {}, '
                                        'an exception occurred : {}.'
                                        .format(key, param.size(), input_param.size(), ex.args))
                else:
                    # 否则在子模块中
                    if isinstance(param, TorchModule):
                        # torch模块
                        # 由于paddle没有提供类似strict的参数，因此也不对torch作要求
                        param.load_state_dict(input_param, strict=False)
                    elif isinstance(param, PaddleLayer):
                        # paddle模块
                        param.load_dict(input_param)
            else:
                missing_keys.append(key)

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))

    def attributes(self, *types):
        """
        查找对应类型的成员
        """
        for name, value in vars(self).items():
            if isinstance(value, types):
                yield name, value

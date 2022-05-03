import os
from typing import Optional, Dict, Union, Callable, Tuple

from .paddle_driver import PaddleDriver
from .utils import replace_batch_sampler, replace_sampler
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.env import USER_CUDA_VISIBLE_DEVICES
from fastNLP.core.utils import (
    auto_param_call,
    get_device_from_visible,
    get_paddle_gpu_str,
    get_paddle_device_id,
)
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.core.samplers import (
    ReproducibleBatchSampler,
    ReproduceBatchSampler,
    ReproducibleSampler,
    RandomSampler,
    re_instantiate_sampler,
)
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import DataParallel
    from paddle.fluid.reader import _DatasetKind

__all__ = [
    "PaddleSingleDriver",
]

class PaddleSingleDriver(PaddleDriver):
    """
    支持 paddle cpu 或单卡 gpu 训练的 driver
    """
    def __init__(self, model, device: Union[str, int], fp16: Optional[bool] = False, **kwargs):
        if isinstance(model, DataParallel):
            raise ValueError("`paddle.DataParallel` is not supported in `PaddleSingleDriver`")

        cuda_visible_devices = os.environ.get(USER_CUDA_VISIBLE_DEVICES, None)
        if cuda_visible_devices == "":
            device = "cpu"
            logger.info("You have set `CUDA_VISIBLE_DEVICES` to '' in system environment variable, and we are gonna to"
                        "use `cpu` instead of `gpu` device.")

        super(PaddleSingleDriver, self).__init__(model, fp16=fp16, **kwargs)

        if device is None:
            raise ValueError("Parameter `device` can not be None in `PaddleSingleDriver`.")

        if device != "cpu":
            if isinstance(device, int):
                device_id = device
            else:
                device_id = get_paddle_device_id(device)
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[USER_CUDA_VISIBLE_DEVICES].split(",")[device_id]
        self.model_device = get_paddle_gpu_str(device)

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def setup(self):
        r"""
        该函数用来初始化训练环境，用于设置当前训练的设备，并将模型迁移到对应设备上。
        """
        device = get_device_from_visible(self.model_device, output_type=str)
        paddle.device.set_device(device)
        self.model.to(device)

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        """
        通过调用 `fn` 来实现训练时的前向传播过程；
        注意 Trainer 和 Evaluator 会调用该函数来实现网络的前向传播过程，其中传入该函数的参数 `fn` 是函数 `get_model_call_fn` 所返回的
        函数；

        :param batch: 当前的一个 batch 的数据；可以为字典或者其它类型；
        :param fn: 调用该函数进行一次计算。
        :param signature_fn: 由 Trainer 传入的用于网络前向传播一次的签名函数，因为当 batch 是一个 Dict 的时候，我们会自动调用 auto_param_call
        函数，而一些被包裹的模型需要暴露其真正的函数签名，例如 DistributedDataParallel 的调用函数是 forward，但是需要其函数签名为 model.module.forward；
        :return: 返回由 `fn` 返回的结果（应当为一个 dict 或者 dataclass，但是不需要我们去检查）；
        """
        if isinstance(batch, Dict) and not self.wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        """
        该函数会接受 Trainer 的 train_fn 或者 Evaluator 的 evaluate_fn，返回一个实际用于调用 driver.model_call 时传入的函数参数；
        该函数会在 Trainer 和 Evaluator 在 driver.setup 函数之后调用；

        之所以设置该函数的目的在于希望将具体的 model_call function 从 driver 中抽离出来，然后将其附着在 Trainer 或者 Evaluator 身上；
        这样是因为在新版的设计中，使用 model 的哪种方法来进行 `train step` 或者 `evaluate step` 是通过额外的参数 `train_fn` 和
         `evaluate_fn` 来确定的，而二者又分别是通过 Trainer 和 Evaluator 来控制的；因此不能将确定具体的 `train step fn` 和
         `evaluate step fn` 的逻辑放在每一个 driver 的初始化的时候（因此在 Trainer 初始化第一个 driver 时，Evaluator 还没有初始化，但是
         `evaluate step fn` 的确定却需要 Evaluator 的初始化），因此我们将这一逻辑抽象到这一函数当中；

        这一函数应当通过参数 `fn` 来判断应当返回的实际的调用的函数，具体逻辑如下所示：
            1. 如果 fn == "train_step" or "evaluate_step"，那么对传入的模型进行检测，如果模型没有定义方法 `fn`，则默认调用模型的 `forward`
             函数，然后给出 warning；
            2. 如果 fn 是其他字符串，那么如果模型没有定义方法 `fn` 则直接报错；
        注意不同的 driver 需要做额外的检测处理，例如在 DDPDriver 中，当传入的模型本身就是 DistributedDataParallel 中，我们只能调用模型的
         forward 函数，因此需要额外的 warning；这一点特别需要注意的问题在于 driver 自己在 setup 时也会对模型进行改变（DDPDriver），因此
         可能需要额外标记最初传入 driver 的模型是哪种形式的；

        :param fn: 应当为一个字符串，该函数通过该字符串判断要返回模型的哪种方法；
        :return: 返回一个元组，包含两个函数，用于在调用 driver.model_call 时传入；
        """
        if hasattr(self.model, fn):
            fn = getattr(self.model, fn)
            if not callable(fn):
                raise RuntimeError(f"The `{fn}` attribute is not `Callable`.")
            logger.debug(f'Use {_get_fun_msg(fn, with_fp=False)}...')
            return fn, None
        elif fn in {"train_step", "evaluate_step"}:
            logger.debug(f'Use {_get_fun_msg(self.model.forward, with_fp=False)}...')
            return self.model, self.model.forward
        else:
            raise RuntimeError(f"There is no `{fn}` method in your {type(self.model)}.")

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler]=None,
                                  reproducible: bool = False):
        r"""
        根据输入的 dataloader 得到一个 支持分布式 （distributed） 与 可复现的 (reproducible) 的 dataloader。

        :param dataloader: 根据 dataloader 设置其对应的分布式版本以及可复现版本
        :param dist: 应当为一个字符串，其值应当为以下之一：[None, "dist", "unrepeatdist"]；为 None 时，表示不需要考虑当前 dataloader
            切换为分布式状态；为 'dist' 时，表示该 dataloader 应该保证每个 gpu 上返回的 batch 的数量是一样多的，允许出现少量 sample ，在
            不同 gpu 上出现重复；为 'unrepeatdist' 时，表示该 dataloader 应该保证所有 gpu 上迭代出来的数据合并起来应该刚好等于原始的
            数据，允许不同 gpu 上 batch 的数量不一致。其中 trainer 中 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "dist"；
            否则为 None ，evaluator 中的 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "unrepeatdist"，否则为 None；
        注意当 dist 为 ReproducibleSampler, ReproducibleBatchSampler 时，是断点重训加载时 driver.load 函数在调用；
        当 dist 为 str 或者 None 时，是 trainer 在初始化时调用该函数；

        :param reproducible: 如果为 False ，不要做任何考虑；如果为 True ，需要保证返回的 dataloader 可以保存当前的迭代状态，使得
            可以可以加载。
        :return: 应当返回一个被替换 sampler 后的新的 dataloader 对象 (注意此处一定需要返回一个新的 dataloader 对象) ；此外，
            如果传入的 dataloader 中是 ReproducibleSampler 或者 ReproducibleBatchSampler 需要重新初始化一个放入返回的
            dataloader 中。如果 dist 为空，且 reproducible 为 False，可直接返回原对象。
        """

        # 暂时不支持iterableDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                    "FastNLP does not support `IteratorDataset` now."
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleIterator 说明是在断点重训时 driver.load 函数调用；
        if isinstance(dist, ReproducibleBatchSampler):
            return replace_batch_sampler(dataloader, dist)
        elif isinstance(dist, ReproducibleSampler):
            return replace_sampler(dataloader, dist)

        # 如果 dist 为 str 或者 None，说明是在 trainer 初试化时调用；
        args = self.get_dataloader_args(dataloader)
        if isinstance(args.batch_sampler, ReproducibleBatchSampler):
            batch_sampler = re_instantiate_sampler(args.batch_sampler)
            return replace_batch_sampler(dataloader, batch_sampler)
        elif isinstance(args.sampler, ReproducibleSampler):
            sampler = re_instantiate_sampler(args.sampler)
            return replace_sampler(dataloader, sampler)

        if reproducible:
            if isinstance(args.sampler, paddle.io.RandomSampler):
                # 如果本来就是随机的，直接替换
                sampler = RandomSampler(args.sampler.data_source)
                logger.debug("Replace paddle RandomSampler into fastNLP RandomSampler.")
                return replace_sampler(dataloader, sampler)
            else:
                batch_sampler = ReproduceBatchSampler(
                    batch_sampler=args.batch_sampler,
                    batch_size=args.batch_size,
                    drop_last=args.drop_last
                )
                return replace_batch_sampler(dataloader, batch_sampler)
        else:
            return dataloader

    def unwrap_model(self):
        if isinstance(self.model, paddle.DataParallel):
            return self.model._layers
        else:
            return self.model

    @property
    def data_device(self):
        """
        单卡模式不支持 data_device；
        """
        return self.model_device

    def is_distributed(self):
        return False

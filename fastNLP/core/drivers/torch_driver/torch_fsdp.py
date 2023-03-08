import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from fastNLP.core.drivers.torch_driver.utils import (DummyGradScaler,
                                                     _DDPWrappingModel)
from fastNLP.core.log import logger
from fastNLP.core.utils import (check_user_specific_params,
                                insert_rank_to_filename)
from fastNLP.envs import (FASTNLP_CHECKPOINT_FILENAME,
                          FASTNLP_DISTRIBUTED_CHECK, FASTNLP_MODEL_FILENAME,
                          rank_zero_call)
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _TORCH_GREATER_EQUAL_1_12
from .ddp import TorchDDPDriver

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp import (FullStateDictConfig,
                                        FullyShardedDataParallel as FSDP,
                                        StateDictType)

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

FASTNLP_FSDP_OPTIM_FILENAME = 'fastnlp_fsdp_optim.pkl.tar'
"""
参考文档：
1. https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
2. https://pytorch.org/docs/stable/fsdp.html?highlight=fsdp
3. https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
4. https://engineering.fb.com/2021/07/15/open-source/fsdp/
"""


class TorchFSDPDriver(TorchDDPDriver):
    r"""
    实现对于 pytorch 自己实现的 ``fully sharded data parallel``；请阅读
    `该文档 <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict>`_
    了解更多。

    .. note::

        ``TorchFSDPDriver`` 大部分行为与 :class:`.TorchDDPDriver` 相同，如果您不
        了解 DDP 的过程，可以查看 :class:`.TorchDDPDriver` 的文档。

    :param model: 传入给 :class:`.Trainer` 的 ``model`` 参数
    :param parallel_device: 用于分布式训练的 ``gpu`` 设备
    :param is_pull_by_torch_run: 标志当前的脚本的启动是否由 ``python -m torch.
        distributed.launch`` 启动的
    :param fp16: 是否开启 fp16 训练
    :param torch_kwargs:

        * *fsdp_kwargs* --
        * *set_grad_to_none* -- 是否在训练过程中在每一次 optimizer 更新后将 grad
          置为 ``None``
        * *non_blocking* -- 表示用于 :meth:`torch.Tensor.to` 方法的参数
          non_blocking
        * *gradscaler_kwargs* -- 用于 ``fp16=True`` 时，提供给 :class:`torch.\
          amp.cuda.GradScaler` 的参数
    :kwargs:
        * *model_wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的
          ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为

        .. note::

            关于该参数的详细说明，请参见 :class:`.Trainer` 和 :func:`~fastNLP.\
            core.auto_param_call`。

        * *output_from_new_proc* (``str``) -- 应当为一个字符串，表示在多进程的
          driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一： ``["all",
          "ignore", "only_error"]``，分别代表 *全部输出*、 *全部忽略* 和 *仅输出错
          误* ，而 rank0 的 **所有信息** 都将被打印出来；当该参数的值不是以上值时，
          该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到 log 文件
          中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为
          ``"only_error"``；

    """

    def __init__(self,
                 model,
                 parallel_device: Optional[Union[List['torch.device'],
                                                 'torch.device']],
                 is_pull_by_torch_run: bool = False,
                 fp16: bool = False,
                 torch_kwargs: Optional[Dict] = None,
                 **kwargs):

        # 在加入很多东西后，需要注意这里调用 super 函数的位置；
        super(TorchDDPDriver, self).__init__(
            model, fp16=fp16, torch_kwargs=torch_kwargs, **kwargs)

        if isinstance(model, torch.nn.DataParallel):
            raise ValueError(
                'Parameter `model` can not be `DataParallel` in '
                '`TorchFSDPDriver`, it should be `torch.nn.Module` '
                'or `torch.nn.parallel.DistributedDataParallel` type.')

        # 如果用户自己在外面初始化 DDP，那么其一定是通过
        #  python -m torch.distributed.launch 拉起的；
        self.is_pull_by_torch_run = is_pull_by_torch_run
        self.parallel_device = parallel_device
        if not is_pull_by_torch_run and parallel_device is None:
            raise ValueError(
                'Parameter `parallel_device` can not be None when using '
                '`TorchFSDPDriver`. This error is caused when your value of '
                'parameter `device` is `None` in your `Trainer` instance.')

        # 注意我们在 initialize_torch_driver 中的逻辑就是如果是
        # is_pull_by_torch_run，那么我们就直接把 parallel_device
        # 置为当前进程的gpu；
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            # 我们的 model_device 一定是 torch.device，而不是一个 list；
            self.model_device = parallel_device[  # type: ignore
                self.local_rank]

        # 如果用户自己在外面初始化了 FSDP；
        self.outside_ddp = False
        if dist.is_initialized() and \
                FASTNLP_DISTRIBUTED_CHECK not in os.environ and \
                'fastnlp_torch_launch_not_ddp' not in os.environ:
            # 如果用户自己在外面初始化了 DDP，那么我们要求用户传入的模型一定是已经
            # 由 DistributedDataParallel 包裹后的模型；
            if not isinstance(model, FSDP):
                raise RuntimeError(
                    'It is not allowed to input a normal model instead of '
                    '`FullyShardedDataParallel` when you initialize the ddp '
                    'process out of our control.')
            if isinstance(model, DistributedDataParallel):
                logger.warning(
                    'You are using `TorchFSDPDriver`, but you have '
                    'initialized your model as `DistributedDataParallel`, '
                    'which will make the `FullyShardedDataParallel` not work '
                    'as expected. You could just delete '
                    '`DistributedDataParallel` wrap operation.')

            self.outside_ddp = True
            # 用户只有将模型上传到对应机器上后才能用 DistributedDataParallel 包
            # 裹，因此如果用户在外面初始化了 DDP，那么在 TorchDDPDriver 中我们就
            # 直接将 model_device 置为 None；
            self.model_device = None

        # 当用户自己在外面初始化 DDP 时我们会将 model_device 置为 None，这时用户
        # 可以通过 `data_device` 将对应的数据移到指定的机器上;
        self._data_device = kwargs.get('data_device', None)
        if isinstance(self._data_device, int):
            if self._data_device < 0:
                raise ValueError(
                    'Parameter `data_device` can not be smaller than 0.')
            _could_use_device_num = torch.cuda.device_count()
            if self._data_device >= _could_use_device_num:
                raise ValueError(
                    'The gpu device that parameter `device` specifies is '
                    'not existed.')
            self._data_device = torch.device(f'cuda:{self._data_device}')
        elif isinstance(self._data_device, str):
            self._data_device = torch.device(self._data_device)
        elif self._data_device is not None and not isinstance(
                self._data_device, torch.device):
            raise ValueError(
                'Parameter `device` is wrong type, please check our '
                'documentation for the right use.')

        self._master_port = None
        # world_size 表示的就是全局的显卡的数量；
        # int(os.environ.get("WORLD_SIZE"))  len(self.parallel_device)
        self.world_size = 0
        self.global_rank = 0

        self._fsdp_kwargs = self._torch_kwargs.get('fsdp_kwargs', {})

        check_user_specific_params(self._fsdp_kwargs, FSDP.__init__,
                                   FSDP.__name__)
        if 'cpu_offload' in self._fsdp_kwargs and kwargs[
                'accumulation_steps'] != 1:
            logger.warning(
                'It is not supported ``accumulation_steps`` when using '
                '``cpu_offload`` in ``FullyShardedDataParallel``.')

        self.output_from_new_proc = kwargs.get('output_from_new_proc',
                                               'only_error')
        assert isinstance(self.output_from_new_proc, str), \
            'Parameter `output_from_new_proc` can only be `str` type.'
        if self.output_from_new_proc not in {'all', 'ignore', 'only_error'}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(
                self.output_from_new_proc)

        # 设置这一参数是因为 evaluator 中也会进行 setup 操作，
        # 但是显然是不需要的也不应该的；
        self._has_setup = False
        # 判断传入的模型是否经过 _has_ddpwrapped 包裹；
        self._has_ddpwrapped = False
        # sync_bn 表示是否要将模型中可能存在的 BatchNorm 层转换为可同步所有卡数据
        # 计算均值和方差的 SyncBatchNorm 层；
        # TODO 暂时设置为 False，FSDP 关于该方面的设置需要研究一下
        self.sync_bn = False

    def configure_ddp(self):
        torch.cuda.set_device(self.model_device)
        if not isinstance(self.model, FSDP):
            self.model = FSDP(
                # 注意这里的 self.model_device 是 `torch.device` type，
                # 因此 self.model_device.index；
                _DDPWrappingModel(self.model),
                device_id=self.model_device.index,
                **self._fsdp_kwargs)

            # 必须先使用 FullyShardedDataParallel 包裹模型后再使用 optimizer
            # 包裹模型的参数，因此这里需要将 optimizer 重新初始化一遍；
            for i in range(len(self.optimizers)):
                self.optimizers[i] = type(self.optimizers[i])(
                    self.model.parameters(), **self.optimizers[i].defaults)

            self._has_ddpwrapped = True

    def unwrap_model(self):
        r"""获取原本模型。

        注意该函数因为需要在特定的时候进行调用，例如 ddp 在 get_model_call_fn
        的时候，因此不能够删除；如果您使用该函数来获取原模型的结构信息，是可以的；但是
        如果您想要通过该函数来获取原模型实际的参数，是不可以的，因为在
        FullyShardedDataParallel 中模型被切分成了多个部分，而对于每个 gpu 上 的模
        型只是整体模型的一部分。"""
        try:
            _module = self.model.module.module
        except AttributeError:
            # 在 torch1.12 中，包裹顺序为 FSDP -> FlattenParamsWrapper
            # -> DDPWrapping，而在 torch1.13 中 FSDP 下就直接是我们的 DDPWrapping
            # 故使用 try-except 来处理
            _module = self.model.module
        except BaseException as e:
            raise e
        if isinstance(_module, _DDPWrappingModel):
            return _module.model
        else:
            return _module

    def save_model(  # type: ignore[override]
            self,
            filepath: Union[str, Path],
            only_state_dict: bool = True,
            **kwargs):
        """保存模型到 ``filepath`` 中。

        :param filepath: 文件路径
        :param only_state_dict: 是否只保存权重；在 ``TorchFSDPDriver`` 中只能为
            ``True``。
        :kwargs:
            * *on_rank0* (``bool``) -- 是否将所有 rank 上的模型参数全部聚合到
              rank0 上，注意这样可能会造成 OOM，``on_rank0`` 默认为 ``False``。
        :return:
        """
        if not only_state_dict:
            raise RuntimeError(
                'When using `TorchFSDPDriver`, only `only_state_dict=True` '
                'is allowed.')
        on_rank0 = kwargs.get('on_rank0', False)
        filepath = Path(filepath)
        if on_rank0:
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model,
                                      StateDictType.FULL_STATE_DICT,
                                      full_state_dict_config):
                state_dict = self.model.state_dict()
            rank_zero_call(torch.save)(state_dict, filepath)
        else:
            # 添加 'rank0/1' 字段来区分全部聚集到 rank0 保存的方式；
            filepath = insert_rank_to_filename(filepath)
            with FSDP.state_dict_type(self.model,
                                      StateDictType.LOCAL_STATE_DICT):
                state_dict = self.model.state_dict()
            torch.save(state_dict, filepath)

    def load_model(  # type: ignore[override]
            self,
            filepath: Union[Path, str],
            only_state_dict: bool = True,
            **kwargs):
        """从 ``filepath`` 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重；在 ``TorchFSDPDriver`` 中
            只能为 ``True``。
        :kwargs:
            * *on_rank0* (``bool``) -- 加载的权重是否是聚合了所有 rank 的权重。
              ``on_rank0`` 默认为 ``False``。
        :return:
        """
        if only_state_dict is False:
            raise RuntimeError(
                'When using `TorchFSDPDriver`, only `only_state_dict=True` '
                'is allowed.')
        on_rank0 = kwargs.get('on_rank0', False)
        filepath = Path(filepath)

        if not on_rank0:
            filepath = insert_rank_to_filename(filepath)
            states = torch.load(filepath)
        else:
            states = torch.load(filepath, map_location='cpu')

        if isinstance(states, dict) and only_state_dict is False:
            logger.rank_zero_warning(
                f'It seems like that {filepath} only contains state, you may '
                'need to use `only_state_dict=True`')
        elif not isinstance(states, dict) and only_state_dict is True:
            logger.rank_zero_warning(
                f'It seems like that {filepath} is not state, you may need to '
                'use `only_state_dict=False`')
        if not isinstance(states, Mapping):
            states = states.state_dict()

        if on_rank0:
            with FSDP.state_dict_type(self.model,
                                      StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(states)
        else:
            with FSDP.state_dict_type(self.model,
                                      StateDictType.LOCAL_STATE_DICT):
                self.model.load_state_dict(states)

    def save_checkpoint(self,
                        folder: Path,
                        states: Dict,
                        dataloader,
                        only_state_dict: bool = True,
                        should_save_model: bool = True,
                        **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存 **优化器** 和 **sampler** 的状态，以及
        **模型** （若 ``should_save_model`` 为 ``True``）

        :param folder: 保存断点重训的状态的文件夹；:meth:`save_checkpoint` 函数应
            该在该路径下面下面新增名为 ``FASTNLP_CHECKPOINT_FILENAME`` 与
            ``FASTNLP_MODEL_FILENAME`` （若 ``should_save_model`` 为 ``True``）
            的文件。把 model 相关的内容放入到 ``FASTNLP_MODEL_FILENAME`` 文件中，
            将传入的 ``states`` 以及自身产生的其它状态一并保存在
            ``FASTNLP_CHECKPOINT_FILENAME`` 里面。
        :param states: 由 :class:`.Trainer` 传入的一个字典，其中已经包含了为了实现
            断点重训所需要保存的其它对象的状态。
        :param dataloader: 正在使用的 dataloader。
        :param only_state_dict: 是否只保存模型的参数。在 ``TorchFSDPDriver`` 中该
            参数仅能为 ``True``。
        :param should_save_model: 是否应该保存模型，如果为 ``False``，Driver 将不
            负责 model 的保存。
        :kwargs:
            * *on_rank0* (``bool``) -- 保存模型和优化器时是否将权重都聚合到 rank0
              上。可能会导致 OOM，``on_rank0`` 默认为 ``False``。
        """
        if not only_state_dict:
            raise RuntimeError(
                'When using `TorchFSDPDriver`, only `only_state_dict=True` '
                'is allowed.')

        on_rank0 = kwargs.get('on_rank0', False)

        # 1. sampler 的状态；
        num_consumed_batches = states.pop('num_consumed_batches')
        states['sampler_states'] = self.get_sampler_state(
            dataloader, num_consumed_batches)

        # 2. 保存模型的状态；
        if should_save_model:
            os.makedirs(folder, exist_ok=True)
            self.barrier()
            model_path = folder.joinpath(FASTNLP_MODEL_FILENAME)
            self.save_model(
                model_path, only_state_dict=only_state_dict, on_rank0=on_rank0)

        # 3. 保存 optimizers 的状态；
        optim_path = folder.joinpath(FASTNLP_FSDP_OPTIM_FILENAME)
        self.save_optimizer(optim_path, on_rank0=on_rank0)
        logger.debug('Save optimizer state dict.')

        # 4. 保存fp16的状态
        if not isinstance(self.grad_scaler, DummyGradScaler):
            grad_scaler_state_dict = self.grad_scaler.state_dict()
            states['grad_scaler_state_dict'] = grad_scaler_state_dict

        # 确保只有 rank0 才会执行实际的保存操作；
        rank_zero_call(torch.save)(
            states, Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME))

    def load_checkpoint(  # type: ignore[override]
            self,
            folder: Path,
            dataloader,
            only_state_dict: bool = True,
            should_load_model: bool = True,
            **kwargs) -> Dict:
        r"""
        断点重训的加载函数，该函数会负责读取数据，并且恢复 **优化器** 、**sampler**
        的状态和 **模型** （如果 ``should_load_model`` 为 True）以及其它在
        :meth:`save_checkpoint` 函数中执行的保存操作，然后将一个 state 字典返回给
        :class:`.Trainer` 字典的内容为函数 :meth:`save_checkpoint` 接受到的
        ``states`` ）。

        该函数应该在所有 rank 上执行。

        :param folder: 读取该 folder 下的 ``FASTNLP_CHECKPOINT_FILENAME`` 文件
            与 ``FASTNLP_MODEL_FILENAME`` （如果 should_load_model 为True）。
        :param dataloader: 当前给定 dataloader，需要根据保存的 dataloader 状态合
            理设置。若该值为 ``None``，则不需要返回 ``'dataloader'`` 以及
            ``'batch_idx_in_epoch'`` 这两个值。
        :param only_state_dict: 是否仅读取模型的 state_dict ，当
            ``should_save_model`` 为 ``False``，该参数无效。如果为 ``True``，说明
            保存的内容为权重；如果为 ``False`` 说明保存的是模型，但也是通过当前
            Driver 的模型去加载保存的模型的权重，而不是使用保存的模型替换当前模型。
        :param should_load_model: 是否应该加载模型，如果为 ``False``，Driver 将不
            负责加载模型。若该参数为 ``True``，但在保存的状态中没有找到对应的模型状
            态，则报错。
        :kwargs:
            * *on_rank0* (``bool``) -- 加载的模型和优化器权重是否是已经全部聚合到
              rank0 的权重，``on_rank0`` 默认为 ``False``。
            * *optim_shard_strategy* (``str``) -- 加载 ``optimizers`` 的状态时，
              决定如何分发 ``state_dict`` 的策略，仅当 ``on_rank0`` 为 ``False``
              时有效。默认为 ``shard``：

              - 为 ``shard`` 时，每个 rank 会先加载完整的 ``state_dict``，然后各自
                进行分片，需要一定存储空间，即 `FSDP.shard_full_optim_state_dict
                <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.\
                fsdp.FullyShardedDataParallel.shard_full_optim_state_dict>`_
              - 为 ``scatter`` 时，仅 rank0 会加载完整的 ``state_dict``，然后分发
                至各个 rank，需要通信成本，即 `FSDP.scatter_full_optim_state_dict
                <https://pytorch.org/docs/stable/fsdp.html#torch.distributed.\
                fsdp.FullyShardedDataParallel.scatter_full_optim_state_dict>`_
        :return: :meth:`save_checkpoint` 函数输入的 ``states`` 内容。除此之外，还
            返回的内容有：

            * *dataloader* -- 根据传入的 ``dataloader`` 与读取出的状态设置为合理状
              态的 dataloader。在当前 ``dataloader`` 样本数与读取出的 sampler 样
              本数不一致时报错。
            * *batch_idx_in_epoch* -- :class:`int` 类型的数据，表明当前 epoch 进
              行到了第几个 batch 。请注意，该值不能仅通过保存的数据中读取的，因为前
              后两次运行的 ``batch_size`` 可能有变化，而应该符合以下等式::

                返回的 dataloader 还会产生的 batch 数量 + batch_idx_in_epoch =
                原来不断点训练时的 batch 的总数

              由于 **返回的 dataloader 还会产生的 batch 数** 在 ``batch_size``
              与 ``drop_last`` 参数给定的情况下无法改变，因此只能通过调整
              ``batch_idx_in_epoch`` 这个值来使等式成立。一个简单的计算原则如下：

              * drop_last 为 ``True`` 时，等同于 floor(sample_in_this_rank/
                batch_size) - floor(num_left_samples/batch_size)；
              * drop_last 为 ``False`` 时，等同于 ceil(sample_in_this_rank/
                batch_size) - ceil(num_left_samples/batch_size)。
        """
        if not only_state_dict:
            raise RuntimeError(
                'When using `TorchFSDPDriver`, only `only_state_dict=True` '
                'is allowed.')
        on_rank0 = kwargs.get('on_rank0', False)
        optim_shard_strategy = kwargs.get('optim_shard_strategy', 'strategy')

        states = torch.load(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME))

        # 1. 加载 optimizers 的状态；
        self.load_optimizer(
            folder.joinpath(FASTNLP_FSDP_OPTIM_FILENAME), on_rank0,
            optim_shard_strategy)

        # 2. 加载模型状态；
        if should_load_model:
            self.load_model(
                filepath=folder.joinpath(FASTNLP_MODEL_FILENAME),
                only_state_dict=only_state_dict,
                on_rank0=on_rank0)

        # 3. 加载 fp16 的状态
        if 'grad_scaler_state_dict' in states:
            grad_scaler_state_dict = states.pop('grad_scaler_state_dict')
            if not isinstance(self.grad_scaler, DummyGradScaler):
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
                logger.debug('Load grad_scaler state dict...')
        elif not isinstance(self.grad_scaler, DummyGradScaler):
            logger.rank_zero_warning(
                f'Checkpoint {folder} is not trained with fp16=True, '
                'while resume to a fp16=True training, the training '
                'process may be unstable.')

        # 4. 恢复 sampler 的状态；
        sampler_states = states.pop('sampler_states')
        states_ret = self.load_sampler_state(dataloader, sampler_states)
        states.update(states_ret)

        return states

    def save_optimizer(self, filepath: Path, on_rank0=False):
        state_dict = self.get_optimizer_state(on_rank0)
        if on_rank0:
            rank_zero_call(torch.save)(state_dict, filepath)
        else:
            torch.save(state_dict, insert_rank_to_filename(filepath))

    def load_optimizer(self, filepath: Path, on_rank0=False, strategy='shard'):
        if not on_rank0:
            filepath = insert_rank_to_filename(filepath)
        states = torch.load(filepath)
        self.load_optimizer_state(states, on_rank0, strategy)

    def get_optimizer_state(self, on_rank0=False):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            if on_rank0:
                optimizer_state = FSDP.full_optim_state_dict(
                    self.model, self.optimizers[i])
            else:
                optimizer_state = self.optimizers[i].state_dict()
            optimizers_state_dict[f'optimizer{i}'] = optimizer_state
        return optimizers_state_dict

    def load_optimizer_state(self, states, on_rank0=False, strategy='shard'):
        # strategy: shard or scatter
        assert strategy in {'shard', 'scatter'}
        assert len(states) == len(self.optimizers), \
            f'The number of optimizers is:{len(self.optimizers)}, ' \
            f'while in checkpoint it is:{len(states)}'

        for i in range(len(self.optimizers)):
            optimizer_state = states[f'optimizer{i}']
            if on_rank0:
                if strategy == 'shard':
                    dist_state = FSDP.shard_full_optim_state_dict(
                        optimizer_state, self.model)
                else:  # scatter
                    dist_state = FSDP.scatter_full_optim_state_dict(
                        optimizer_state, self.model)
            else:
                dist_state = optimizer_state
            self.optimizers[i].load_state_dict(dist_state)

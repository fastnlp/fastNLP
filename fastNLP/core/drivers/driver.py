import os
import signal
import sys
from typing import Any, Sequence, List, Optional, Callable, Dict, Union
from abc import ABC
from datetime import datetime
from pathlib import Path
from io import BytesIO

__all__ = [
    'Driver'
]

from fastNLP.core.utils import nullcontext


# todo 航总 check 一下哪一些方法需要 @abstractmethod；
class Driver(ABC):
    r"""
    用来初始化 `Driver` 的基类，所有定制的 `driver` 都需要继承此类；
    fastNLP 提供的 driver 实例都会同时被 Trainer 和 Evaluator 调用；
    """

    def __init__(self, model):
        r"""
        :param model: 训练或者评测的模型，需要注意该模型可能为用户已经使用类似 `torch.nn.DataParallel` 或者
         `torch.nn.parallel.DistributedDataParallel` 包裹过的模型；
        """
        self.model = model

        # 这些属性用于 open_subprocess 和 on_exception 函数协同配合；
        # self._consensus_file: Optional[Union[str, Path]] = None
        self._pids: Optional[List[int]] = None

    def setup(self):
        r"""
        该函数用来初始化训练环境，例如将模型迁移到对应的设备上等；
        多卡的 driver 的该函数要更为复杂一些，例如其可能需要开启多进程之间的通信环境，以及设置一些环境变量和其余所需要的变量值；
        """

    def replace_sampler(self, dataloader, dist_sampler: Optional[str], reproducible: bool = False):
        r"""
        因为一些特殊的情况需要替换 dataloader 的 sampler，而每一个 driver 中的该函数会提供该功能；例如在多卡训练的中，我们
         需要将 sampler 替换为 distributed sampler；以及如果用户在 Trainer 中加入了断点重训的 callback，那么我们就需要将 sampler 替换
         为 reproducible sampler；

        :param dataloader: 由 trainer 中传入的原始的 dataloader；
        :param dist_sampler: 应当为一个字符串，其值应当为以下之一：[None, "dist", "unrepeatdist"]；用于指定使用怎样的 sampler；
         目前该参数被定制为分布式训练服务，其中 trainer 中 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "dist"，否则为 None；
         evaluator 中的 kwargs 的参数 `use_dist_sampler` 为 True 时，该值为 "unrepeatdist"，否则为 None；
        :param reproducible: 用于在 `Trainer` 中指定是否替换为断点重训的 sampler（多卡） 或者 batch_sampler（单卡）；如果是单卡的 Driver，
         并且该参数为 True，表示当前正在断点重训，那么我们就会使用我们的 `ReproducibleBatchSampler` 来替换 dataloader 原本的 batch_sampler；
         如果是多卡的 Driver，那么我们就会用 `RandomSampler` 替换 dataloader 原本的 sampler；

        :return: 应当返回一个被替换 sampler 后的新的 dataloader 对象 (注意此处一定需要返回一个新的 dataloader 对象) ；
        """
        raise NotImplementedError("Each specific driver should implemented its own `replace_sampler` function.")

    def set_deterministic_dataloader(self, dataloader):
        r"""
        为了确定性训练要对 dataloader 进行修改，保证在确定随机数种子后，每次重新训练得到的结果是一样的；例如对于 torch 的 dataloader，其
         需要将 worker_init_fn 替换；
        """

    def set_sampler_epoch(self, dataloader, cur_epoch_idx):
        r"""
        对于分布式的 sampler，例如 torch 的 DistributedSampler，其需要在每一个 epoch 前设置随机数种子，来保证每一个进程上的 shuffle 是一样的；

        :param cur_epoch_idx: 当前是第几个 epoch；
        """

    def train_step(self, batch):
        """
        通过调用模型自带的 `train_step` 或者 `forward` 方法来实现训练的前向过程；
        如果检测到用户模型实现了 train_step

        :param batch: 当前的一个 batch 的数据；可以为字典或者其它类型；
        :return: 返回由模型的 `train_step` 或者 `forward` 方法返回的结果（应当为一个 dict 或者 dataclass，但是不需要我们去检查）；
        """
        raise NotImplementedError("Each specific driver should implemented its own `train_step` function.")

    def validate_step(self, batch):
        """
        通过调用模型自带的 `validate_step` 或者 `forward` 方法来实现模型评测的前向过程；

        :param batch: 当前的一个 batch 的数据；可以为字典或者其它类型；
        :return: 返回由模型的 `validate_step` 或者 `forward` 方法返回的结果（应当为一个 dict 或者 dataclass，但是不需要我们去检查）；
        """
        raise NotImplementedError("Each specific driver should implemented its own `validate_step` function.")

    def test_step(self, batch):
        """
        通过调用模型自带的 `test_step` 或者 `forward` 方法来实现模型评测的前向过程；

        :param batch: 当前的一个 batch 的数据；可以为字典或者其它类型；
        :return: 返回由模型的 `test_step` 或者 `forward` 方法返回的结果（应当为一个 dict 或者 dataclass，但是不需要我们去检查）；
        """
        raise NotImplementedError("Each specific driver should implemented its own `test_step` function.")

    def check_evaluator_mode(self, mode: str):
        r"""
        因为我们在具体的 driver 的 validate_step 和 test_step 的逻辑是如果模型没有实现本函数，那么就去检测模型是否实现了另一个函数；
        因此如果用户的 evaluator mode 是 validate，但是传入的 model 却没有实现 validate_step 函数，而是实现了 test_step 函数，那么
         我们应当提醒用户这一行为；
        """
        raise NotImplementedError("Each specific driver should implemented its own `predict_step` function.")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, train_dataloader: Any):
        self._train_dataloader = train_dataloader

    @property
    def validate_dataloaders(self):
        return self._validate_dataloaders

    @validate_dataloaders.setter
    def validate_dataloaders(self, validate_dataloaders: Any):
        self._validate_dataloaders = validate_dataloaders

    @property
    def test_dataloaders(self):
        return self._test_dataloaders

    @test_dataloaders.setter
    def test_dataloaders(self, test_dataloaders: Any):
        self._test_dataloaders = test_dataloaders

    @property
    def predict_dataloaders(self):
        return self._predict_dataloaders

    @predict_dataloaders.setter
    def predict_dataloaders(self, predict_dataloaders: Any):
        self._predict_dataloaders = predict_dataloaders

    def set_dataloader(self, **kwargs):
        r"""
        设置训练或者检验过程中的数据；用于在 trainer 和 evaluator 中将数据 dataloader 挂载到每一个具体的 driver 上；

        :param kwargs: 输入的数据，应当使用 'keyword-only' 的参数进行设置；
        """
        if "train_dataloader" in kwargs:
            self.train_dataloader = kwargs["train_dataloader"]
            self._check_dataloader_legality(self.train_dataloader, "train_dataloader", is_train=True)
        if "validate_dataloaders" in kwargs:
            self.validate_dataloaders = kwargs["validate_dataloaders"]
            self._check_dataloader_legality(self.validate_dataloaders, "validate_dataloaders", is_train=False)
        if "test_dataloaders" in kwargs:
            self.test_dataloaders = kwargs["test_dataloaders"]
            self._check_dataloader_legality(self.test_dataloaders, "test_dataloaders", is_train=False)
        if "predict_dataloaders" in kwargs:
            self.predict_dataloaders = kwargs["predict_dataloaders"]
            self._check_dataloader_legality(self.predict_dataloaders, "predict_dataloaders", is_train=False)

    @staticmethod
    def _check_dataloader_legality(dataloader, dataloader_name, is_train: bool = False):
        r"""
        该函数会在 trainer 或者 evaluator 设置 dataloader 后检测 dataloader 的合法性，因为不同的深度学习的框架需要的 dataloader 的
         行为是不相同的；

        :param dataloader: 需要检测的输入的 `dataloader`；
        :param dataloader_name:
        """
        raise NotImplementedError("Each specific driver should implemented its own `_check_dataloader_legality` function.")

    def has_train_dataloader(self):
        return "_train_dataloader" in self.__dict__

    def has_validate_dataloaders(self):
        return "_validate_dataloaders" in self.__dict__

    def has_test_dataloaders(self):
        return "_test_dataloaders" in self.__dict__

    def has_predict_dataloaders(self):
        return "_predict_dataloaders" in self.__dict__

    @property
    def optimizers(self) -> List:
        r"""
        如下所示，driver 返回的 optimizers 一定是一个 List，如果用户直接向 Trainer 传入一个单独的 optimzer，我们会使用一个 List 将其
        包裹；

        :return: List[optimizer0, optimizer1, optimizer2, ...]
        """
        return self._optimizers

    @optimizers.setter
    def optimizers(self, optimizers):
        if not isinstance(optimizers, Sequence):
            self._optimizers = [optimizers]
        else:
            self._optimizers = optimizers
        self._check_optimizer_legality(self._optimizers)

    @property
    def model_device(self):
        return self._model_device

    @model_device.setter
    def model_device(self, model_device):
        self._model_device = model_device

    @property
    def data_device(self):
        return self.model_device

    @staticmethod
    def _check_optimizer_legality(optimizers):
        """
        对于用户传入 trainer 的每一个 optimizer，检测其是否合理，因为不同的深度学习框架所使用的的 optimizer 是不相同的；

        :param optimizers: 需要检测的 `optimizers`；
        """
        raise NotImplementedError("Each specific driver should implemented its own `_check_optimizer_legality` function.")

    def set_optimizers(self, optimizers=None):
        """
        trainer 会调用该函数将用户传入的 optimizers 挂载到 driver 实例上；
        :param optimizers:
        :return:
        """
        self.optimizers = optimizers

    def backward(self, loss):
        """
        实现深度学习中的反向传播过程；

        :param loss: 用来实现反向传播的损失函数值；
        """
        raise NotImplementedError("Each specific driver should implemented its own `backward` function.")

    def step(self):
        r"""
        实现深度学习中的参数的优化更新过程，应当直接通过优化器 optimizers 来更新参数；
        """
        raise NotImplementedError("Each specific driver should implemented its own `step` function.")

    def zero_grad(self, set_to_none: bool = False):
        r"""
        实现深度学习中的梯度的置零操作，应当直接通过优化器 optimizers 来将梯度置零；
        注意梯度累积不需要在这里实现，trainer 已经在内部实现了梯度累积；

        :param set_to_none: 用来判断是否需要将梯度直接置为 None；
        """
        raise NotImplementedError("Each specific driver should implemented its own `zero_grad` function.")

    def get_no_sync_context(self):
        r"""
        返回一个用于关闭多进程之间互相同步操作的 context 上下文对象；只有多卡的 driver 需要单独实现该函数，单卡的 driver 不需要；

        :return: 返回一个类似于 DistributedDataParallel(model).no_sync 的 context 上下文对象；
        """
        return nullcontext

    def get_evaluate_context(self):
        r"""
        返回一个不计算梯度的环境用来对模型进行评测；

        :return: 一个类似 `torch.no_grad` 的 context 上下文对象；
        """
        return nullcontext

    @property
    def auto_cast(self):
        """
        fp16 的上下文环境；

        :return: 返回一个用于 fp16 计算的上下文环境；
        """
        return self._auto_cast

    @auto_cast.setter
    def auto_cast(self, auto_cast):
        self._auto_cast = auto_cast

    def save_model(self, filepath: Union[str, Path, BytesIO], only_state_dict: bool = True, **kwargs):
        r"""
        保存模型的函数；注意函数 `save` 是用来进行断点重训的函数；

        :param filepath: 保存文件的文件位置（需要包括文件名）或一个 BytesIO 对象；
        :param only_state_dict: 是否只保存模型的 `state_dict`；
        :param model_save_fn: 用户传入的用来代替该函数本身保存逻辑的函数；如果该参数不为 None，那么我们会调用 model_save_fn(path)；
        """
        raise NotImplementedError("Each specific driver should implemented its own `save_model` function.")

    def load_model(self, filepath: Union[str, Path, BytesIO], only_state_dict: bool = False, **kwargs):
        r"""
        加载模型的函数；将 filepath 中的模型加载并赋值给当前 model 。

        :param filepath: 需要被加载的对象的文件位置（需要包括文件名）或一个 BytesIO 对象；
        :param load_state_dict: 保存的文件是否只是模型的权重，还是完整的模型。即便是保存的完整的模型，此处也只能使用尝试加载filepath
            模型中的权重到自身模型，而不会直接替代当前 Driver 中的模型。
        :return: 返回加载指定文件后的结果；
        """
        raise NotImplementedError("Each specific driver should implemented its own `load_model` function.")

    def save(self, folder, states: Dict, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存模型和 optimizers, fp16 的 state_dict；以及模型的保存（若 should_save_model 为 True）

        :param folder: 保存断点重训的状态的文件夹；save 函数应该在下面新增两（一）个文件 的 FASTNLP_CHECKPOINT_FILENAME 文件与
            FASTNLP_MODEL_FILENAME （如果 should_save_model 为 True ）。把 model 相关的内容放入到 FASTNLP_MODEL_FILENAME 文件
            中，将传入的 states 以及自身产生其它状态一并保存在 FASTNLP_CHECKPOINT_FILENAME 里面。
        :param states: 由 trainer 传入的一个字典，其中已经包含了为了实现断点重训所需要保存的其它对象的状态，Driver 应该只需要保存
            该对象即可， Driver 应该不需要理解该对象，同时在 driver.load() 的时候，需要将 states 返回回去，load() 返回的值与这里的
            传入的值保持一致。
        :param only_state_dict: 是否只保存模型的参数，当 should_save_model 为 False ，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为False，Driver 将不负责 model 的保存。
        """
        raise NotImplementedError("Each specific driver should implemented its own `save` function.")

    def load(self, folder: Union[str, Path], only_state_dict: bool =True, should_load_model: bool = True,  **kwargs) -> Dict:
        r"""
        断点重训的加载函数，注意该函数会负责读取数据，并且恢复 optimizers , fp16 的 state_dict 和 模型（根据 should_load_model ）和；
            其它在 Driver.save() 函数中执行的保存操作，然后将一个 state 字典返回给 trainer （ 内容为Driver.save() 接受到的 states ）。

        该函数应该在所有 rank 上执行。

        :param folder: 读取该 folder 下的 FASTNLP_CHECKPOINT_FILENAME 文件与 FASTNLP_MODEL_FILENAME
            （如果 should_load_model 为True）。
        :param only_state_dict: 读取的，当 should_save_model 为 False ，该参数无效。如果为 True ，说明保存的内容为权重；如果为
            False 说明保存的是模型，但也是通过当前 Driver 的模型去加载保存的模型的权重，而不是使用保存的模型替换当前模型。
        :param should_load_model: 是否应该加载模型，如果为False，Driver 将不负责加载模型。若该参数为 True ，但在保存的状态中没有
            找到对应的模型状态，则报错。
        :return: 需要返回 save 函数输入的 states 内容；
        """
        raise NotImplementedError("Each specific driver should implemented its own `load` function.")

    @staticmethod
    def tensor_to_numeric(tensor, reduce: Optional[str]=None):
        r"""
        将一个 `tensor` 对象（仅处理当前 driver 使用的 tensor 即可）转换为 python 的 `numeric` 对象；如果 tensor 只包含一个
            元素则返回 float 或 int 。

        :param tensor: 需要被转换的 `tensor` 对象
        :param reduce: 可选 ['sum', 'max', 'mea', 'min']，如果不为 None 将使用该 reduce 方法来处理当前 tensor 再返回
            float 或 int 对象。
        :return: 转换后返回的结果
        """
        raise NotImplementedError("Each specific driver should implemented its own `tensor_to_numeric` function.")

    def set_model_mode(self, mode: str):
        r"""
        设置模型为 `train` / `eval` 的模式；目的是为切换模型训练和推理（会关闭dropout等）模式；

        :param mode: 应为二者之一：["train", "eval"]；
        """

    def unwrap_model(self):
        """
        保证用户拿到的模型一定是最原始的模型；
        注意因为我们把保存模型的主要逻辑和代码移到了 `Driver` 中，因此在 `save_model` 函数中，一定要先调用此函数来保证我们保存的模型一定是
         最为原始的模型；
        需要注意用户本身传入的模型就是经过类似 `torch.nn.DataParallel` 或者 `torch.nn.parallel.DistributedDataParallel` 包裹的模型，
        因此在该函数内需要先判断模型的类别；

        :return: 返回最原始的模型，例如没有被 `DistributedDataParallel` 包裹的模型；
        """

    @staticmethod
    def move_model_to_device(model, device):
        r"""
        用来将模型转移到指定的 device 上；
        之所以写成 `staticmethod`，是因为一方面在 `Driver` 中我们要使用 `unwrap_model` 来拿到最原始的模型，另一方面，在 `save_model`
         中，我们需要先将模型移到 cpu 后，又再移到 gpu 上，因此不适宜在该函数内部调用 `unwrap_model`，而是将 model 作为该函数的参数；
        """

    def move_data_to_device(self, batch):
        r"""
        将数据迁移到指定的机器上；batch 可能是 list 也可能 dict ，或其嵌套结构。

        :return: 将移动到指定机器上的 batch 对象返回；
        """

    def get_local_rank(self) -> int:
        r"""
        返回当前的local_rank，本函数的返回值只在运行分布式训练的时候有实际含义。

        :return: 一个整数值，表示当前进程在当前这台机器上的序号；
        """
        return 0

    def barrier(self):
        r"""
        用于在多进程工作时同步各进程的工作进度，运行快的进程运行到这里会等待运行慢的进程，只有所有进程都运行到此函数时，所有的进程才会继续运行；
        仅在多分布式训练场景中有使用。
        """

    @staticmethod
    def get_dataloader_args(dataloader):
        """
        用于从 dataloader 中抽取一些属性的值，返回的dataclass中必须包含以下的key：
            sampler, batch_sampler, batch_size, drop_last；

        :param dataloader:
        :return: 返回一个 dataclass，其实例属性应当包括以上的各个属性，并且其名字也应当与这些属性相同，从而方便 trainer 或者其它对象调用；
        """
        raise NotImplementedError("Each specific driver should implemented its own `get_dataloader_args` function.")

    def is_distributed(self) -> bool:
        """
        当前的 driver 实例是否是分布式的；

        :return: 返回一个 bool 值，如果当前的 driver 实例是用于分布式的，那么返回 True；
        """
        return False

    def on_exception(self):
        """
        该函数用于在训练或者预测过程中出现错误时正确地关掉其它的进程，这一点是通过在多进程 driver 调用 open_subprocess 的时候将每一个进程
         的 pid 记录下来，然后在出现错误后，由出现错误的进程手动地将其它进程 kill 掉；

        因此，每一个多进程 driver 如果想要该函数能够正确地执行，其需要在自己的 open_subprocess（开启多进程的函数）中正确地记录每一个进程的
         pid 的信息；
        """
        # 单卡 driver 不需要这个函数；
        if self._pids is not None:

            exc_type, exc_value, exc_traceback_obj = sys.exc_info()
            _write_exc_info = {
                'exc_type': exc_type,
                'exc_value': exc_value,
                'time': str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                'global_rank': getattr(self, "global_rank", None),
                'rank': self.get_local_rank(),
            }
            sys.stderr.write(str(_write_exc_info)+"\n")

            sys.stderr.write(f"Start to stop these pids:{self._pids}, please wait several seconds.\n")
            for pid in self._pids:
                if pid != os.getpid():
                    os.kill(pid, signal.SIGKILL)

    def broadcast_object(self, obj, src:int=0, group=None, **kwargs):
        """
        从 src 端将 obj 对象（可能是 tensor ，可能是 object ）broadcast 到其它所有进程。如果是非 tensor 的对象会尝试使用 pickle 进行打包进行
            传输，然后再 dst 处再加载回来。仅在分布式的 driver 中有实际意义。

        :param obj: obj，可能是 Tensor 或 嵌套类型的数据
        :param int src: source 的 global rank 。
        :param group: 所属的 group
        :param kwargs:
        :return: 输入的 obj 。
        """
        if not self.is_distributed():
            return obj
        raise NotImplementedError(f"Driver:{self.__class__.__name__} does not support `broadcast_object` method right "
                                  f"now.")

    def all_gather(self, obj, group)->List:
        """
        将 obj 互相传送到其它所有的 rank 上，其中 obj 可能是 Tensor，也可能是嵌套结构的 object 。如果不是基础类型的数据，尝试通过
            pickle 进行序列化，接收到之后再反序列化。

        :param obj: 可以是 float/int/bool/np.ndarray/{}/[]/Tensor等。
        :param group:
        :return: 返回值应该是 [obj0, obj1, ...], 其中obj1是rank0上的对象，obj1是rank1上的对象...
        """
        if not self.is_distributed():
            return [obj]
        raise NotImplementedError(f"Driver:{self.__class__.__name__} does not support `all_gather` method right "
                                  f"now.")


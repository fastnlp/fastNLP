import os
import signal
import sys
from typing import Any, Sequence, List, Optional, Callable, Dict, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from io import BytesIO
import json

__all__ = [
    'Driver'
]

from fastNLP.core.utils import nullcontext


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

    @abstractmethod
    def setup(self):
        r"""
        该函数用来初始化训练环境，例如将模型迁移到对应的设备上等；
        多卡的 driver 的该函数要更为复杂一些，例如其可能需要开启多进程之间的通信环境，以及设置一些环境变量和其余所需要的变量值；
        """

    def set_dist_repro_dataloader(self, dataloader, dist=None, reproducible: bool = False):
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
        if dist is None and reproducible is False:
            return dataloader
        raise NotImplementedError(f"Driver:{self.__class__.__name__} does not support `set_dist_repro_dataloader` "
                                  f"function.")

    def set_deterministic_dataloader(self, dataloader):
        r"""
        为了确定性训练要对 dataloader 进行修改，保证在确定随机数种子后，每次重新训练得到的结果是一样的；例如对于 torch 的 dataloader，其
         需要将 worker_init_fn 替换；
        """

    def set_sampler_epoch(self, dataloader, cur_epoch_idx):
        r"""
        对于分布式的 sampler，例如 torch 的 DistributedSampler，其需要在每一个 epoch 前设置随机数种子，来保证每一个进程上的 shuffle 是一样的；
            dataloader 中可能真正发挥作用的是 batch_sampler 也可能是 sampler。

        :param dataloader: 需要设置 epoch 的 dataloader 。
        :param cur_epoch_idx: 当前是第几个 epoch；
        """

    @abstractmethod
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
        raise NotImplementedError("Each specific driver should implemented its own `model_call` function.")

    @abstractmethod
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
        raise NotImplementedError("Each specific driver should implemented its own `get_model_call_fn` function.")

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizers(self) -> List:
        r"""
        如下所示，driver 返回的 optimizers 一定是一个 List，如果用户直接向 Trainer 传入一个单独的 optimizer，我们会使用一个 List 将其
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

    @abstractmethod
    def backward(self, loss):
        """
        实现深度学习中的反向传播过程；

        :param loss: 用来实现反向传播的损失函数值；
        """
        raise NotImplementedError("Each specific driver should implemented its own `backward` function.")

    @abstractmethod
    def step(self):
        r"""
        实现深度学习中的参数的优化更新过程，应当直接通过优化器 optimizers 来更新参数；
        """
        raise NotImplementedError("Each specific driver should implemented its own `step` function.")

    @abstractmethod
    def zero_grad(self, set_to_none: bool = False):
        r"""
        实现深度学习中的梯度的置零操作，应当直接通过优化器 optimizers 来将梯度置零；
        注意梯度累积不需要在这里实现，trainer 已经在内部实现了梯度累积；

        :param set_to_none: 用来判断是否需要将梯度直接置为 None；
        """
        raise NotImplementedError("Each specific driver should implemented its own `zero_grad` function.")

    def get_model_no_sync_context(self):
        r"""
        返回一个用于关闭多进程之间 model 中的自动互相同步操作的 context 上下文对象；只有多卡的 driver 需要单独实现该函数，
            单卡的 driver 不需要；

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

    @abstractmethod
    def save_model(self, filepath: Union[str, Path, BytesIO], only_state_dict: bool = True, **kwargs):
        r"""
        保存模型的函数；注意函数 `save` 是用来进行断点重训的函数；

        :param filepath: 保存文件的文件位置（需要包括文件名）或一个 BytesIO 对象；
        :param only_state_dict: 是否只保存模型的 `state_dict`；
        :param model_save_fn: 用户传入的用来代替该函数本身保存逻辑的函数；如果该参数不为 None，那么我们会调用 model_save_fn(path)；
        """
        raise NotImplementedError("Each specific driver should implemented its own `save_model` function.")

    @abstractmethod
    def load_model(self, filepath: Union[str, Path, BytesIO], only_state_dict: bool = False, **kwargs):
        r"""
        加载模型的函数；将 filepath 中的模型加载并赋值给当前 model 。

        :param filepath: 需要被加载的对象的文件位置（需要包括文件名）或一个 BytesIO 对象；
        :param load_state_dict: 保存的文件是否只是模型的权重，还是完整的模型。即便是保存的完整的模型，此处也只能使用尝试加载filepath
            模型中的权重到自身模型，而不会直接替代当前 Driver 中的模型。
        :return: 返回加载指定文件后的结果；
        """
        raise NotImplementedError("Each specific driver should implemented its own `load_model` function.")

    @abstractmethod
    def save(self, folder, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存模型和 optimizers, fp16 的 state_dict；以及模型的保存（若 should_save_model 为 True）

        :param folder: 保存断点重训的状态的文件夹；save 函数应该在下面新增两（一）个文件 的 FASTNLP_CHECKPOINT_FILENAME 文件与
            FASTNLP_MODEL_FILENAME （如果 should_save_model 为 True ）。把 model 相关的内容放入到 FASTNLP_MODEL_FILENAME 文件
            中，将传入的 states 以及自身产生其它状态一并保存在 FASTNLP_CHECKPOINT_FILENAME 里面。
        :param states: 由 trainer 传入的一个字典，其中已经包含了为了实现断点重训所需要保存的其它对象的状态，Driver 应该只需要保存
            该对象即可， Driver 应该不需要理解该对象，同时在 driver.load() 的时候，需要将 states 返回回去，load() 返回的值与这里的
            传入的值保持一致。
        :param dataloader: 正在使用的 dataloader，需要保存里面的状态使得之后可以从当前迭代的位置恢复。
        :param only_state_dict: 是否只保存模型的参数，当 should_save_model 为 False ，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为False，Driver 将不负责 model 的保存。
        """
        raise NotImplementedError("Each specific driver should implemented its own `save` function.")

    @abstractmethod
    def load(self, folder: Union[str, Path], dataloader, only_state_dict: bool =True, should_load_model: bool = True,  **kwargs) -> Dict:
        r"""
        断点重训的加载函数，注意该函数会负责读取数据，并且恢复 optimizers , fp16 的 state_dict 和 模型（根据 should_load_model ）和；
            其它在 Driver.save() 函数中执行的保存操作，然后将一个 state 字典返回给 trainer （ 内容为Driver.save() 接受到的 states ）。

        该函数应该在所有 rank 上执行。

        :param folder: 读取该 folder 下的 FASTNLP_CHECKPOINT_FILENAME 文件与 FASTNLP_MODEL_FILENAME
            （如果 should_load_model 为True）。
        :param dataloader: 当前给定 dataloader，需要根据 save 的 dataloader 状态合理设置。若该值为 None ，是不需要返回 'dataloader'
            以及 'batch_idx_in_epoch' 这两个值。
        :param only_state_dict: 读取的，当 should_save_model 为 False ，该参数无效。如果为 True ，说明保存的内容为权重；如果为
            False 说明保存的是模型，但也是通过当前 Driver 的模型去加载保存的模型的权重，而不是使用保存的模型替换当前模型。
        :param should_load_model: 是否应该加载模型，如果为False，Driver 将不负责加载模型。若该参数为 True ，但在保存的状态中没有
            找到对应的模型状态，则报错。
        :return: 需要返回 save 函数输入的 states 内容
            'dataloader'，返回的是根据传入的 dataloader 与 保存的状态一起设置为合理的状态，可以返回的对象与传入的dataloader是同一个。
                在保存与当前传入 data sample 数目不一致时报错。
            'batch_idx_in_epoch': int 类型的数据，表明当前 epoch 进行到了进行到了第几个 batch 了。 请注意，该值不能是只能通过保存的
                数据中读取的，因为前后两次运行 batch_size 可能由变化。该数字的原则应该符合以下等式
                '返回 dataloader 还会产生的batch数量' + 'batch_idx_in_epoch' = '原来不断点训练的batch的总数' 。
                由于 '返回 dataloader 还会产生的batch数量' 这个数量在 batch_size 与 drop_last 参数给定的情况下，无法改变，因此
                只能通过调整 batch_idx_in_epoch 这个值来使等式成立。一个简单的计算原则如下
                当drop_last为True，等同于 floor(sample_in_this_rank/batch_size) - floor(num_left_samples/batch_size);
                当drop_last为False，等同于 ceil(sample_in_this_rank/batch_size) - ceil(num_left_samples/batch_size)。
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

    @abstractmethod
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

    @abstractmethod
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

        注意，该函数的行为会受到 FASTNLP_NO_SYNC 的影响。仅当 FASTNLP_NO_SYNC 在 os.environ 中不存在，或小于 1 时才真的执行 barrier 。
        """

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
                'exc_type': str(exc_type.__name__),
                'exc_value': str(exc_value),
                'exc_time': str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S')),
                'exc_global_rank': getattr(self, "global_rank", None),
                'exc_local_rank': self.get_local_rank(),
            }
            sys.stderr.write("\nException info:\n")
            sys.stderr.write(json.dumps(_write_exc_info, indent=2)+"\n")

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


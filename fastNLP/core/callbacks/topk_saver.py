__all__ = [
    'TopkSaver'
]
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Tuple, Callable, Union

from ...envs.distributed import rank_zero_rm
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_LAUNCH_TIME
from fastNLP.envs import rank_zero_call
from fastNLP.envs.env import FASTNLP_EVALUATE_RESULT_FILENAME
from .has_monitor_callback import ResultsMonitor


class Saver:
    """
    执行保存的对象。保存的文件组织结构为::

        - folder  # 当前初始化的参数
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - folder_name  # 由 save() 调用时传入。

    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param save_object: 可选 ``['trainer', 'model']`` ，表示在保存时的保存对象为 ``trainer+model`` 还是 只是 ``model`` 。如果
        保存 ``trainer`` 对象的话，将会保存 :class:`~fastNLP.core.controllers.Trainer` 的相关状态，可以通过 :meth:`Trainer.load_checkpoint` 加载该断
        点继续训练。如果保存的是 ``Model`` 对象，则可以通过 :meth:`Trainer.load_model` 加载该模型权重。
    :param only_state_dict: 保存时是否仅保存权重，在 model_save_fn 不为 None 时无意义。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函数应当接受一个文件夹作为参数，不返回任何东西。
        如果传入了 model_save_fn 函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运行该函数。
    :param kwargs: 更多需要传递给 Trainer.save_checkpoint() 或者 Trainer.save_model() 接口的参数。
    """
    def __init__(self, folder:str=None, save_object:str='model', only_state_dict:bool=True,
                 model_save_fn:Callable=None, **kwargs):
        if folder is None:
            folder = Path.cwd().absolute()
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        elif folder.is_file():
            raise ValueError("Parameter `folder` should be a directory instead of a file.")

        self.folder = folder
        self.only_state_dict = only_state_dict
        self.model_save_fn = model_save_fn
        self.kwargs = kwargs
        self.save_object = save_object
        self.save_fn_name = 'save_checkpoint' if save_object == 'trainer' else 'save_model'

        self.timestamp_path = self.folder.joinpath(os.environ[FASTNLP_LAUNCH_TIME])

    @rank_zero_call
    def save(self, trainer, folder_name):
        """
        执行保存的函数，将数据保存在::

            - folder/
                - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                    - folder_name  # 当前函数参数

        :param trainer: Trainer 对象
        :param folder_name: 保存的 folder 名称，将被创建。
        :return: 返回实际发生保存的 folder 绝对路径。如果为 None 则没有创建。
        """
        folder = self.timestamp_path.joinpath(folder_name)
        folder.mkdir(parents=True, exist_ok=True)
        save_fn = getattr(trainer, self.save_fn_name)
        save_fn(
            folder=folder,
            only_state_dict=self.only_state_dict,
            model_save_fn=self.model_save_fn,
            **self.kwargs
        )
        return str(os.path.abspath(folder))

    @rank_zero_call
    def save_json(self, results, path):
        """
        以 json 格式保存 results 到 path 中

        :param results: 一般是评测后的结果。
        :param path: 保存的文件名
        :return:
        """
        with open(path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2)

    @rank_zero_call
    def rm(self, folder_name):
        """
        移除 folder/timestamp/folder_name 。其中 folder 为用户在初始化指定, timestamp 为当前脚本的启动时间。

        :param folder_name: 需要移除的路径。
        :return:
        """
        folder = self.timestamp_path.joinpath(folder_name)
        rank_zero_rm(folder)

    def state_dict(self):
        states = {
            'timestamp_path': str(self.timestamp_path),
        }
        return states

    def load_state_dict(self, states):
        timestamp_path = states['timestamp_path']
        if not os.path.exists(timestamp_path):
            logger.info(f"The resuming checkpoint folder {timestamp_path} is not exists, checkpoint will save to "
                        f" {self.timestamp_path.absolute()}.")
        else:
            logger.info(f"Resume to save checkpoint in path: {timestamp_path}.")
            self.timestamp_path = Path(timestamp_path)

    def __str__(self):
        return f'saver:{self.save_object}'


class TopkQueue:
    """
    用于维护处于 topk 的 key, value 对。

    :param int topk: 整数，-1 表示所有数据都是 topk 的; 如果是 0, 表示没有任何数据是满足 topk 的。
    """
    def __init__(self, topk):
        assert isinstance(topk, int)
        self.topk = topk
        self.topk_dict = {}  # 其中 key 为保存的内容， value 是对应的性能。

    def push(self, key, value) -> Optional[Tuple[Union[str, None], Union[float, None]]]:
        """
        将 key/value 推入 topk 的 queue 中，以 value 为标准，如果满足 topk 则保留此次推入的信息，同时如果新推入的数据将之前的数据给
        挤出了 topk ，则会返回被挤出的 (key, value)；如果返回为 (None, None)，说明满足 topk 且没有数据被挤出。如果不满足 topk ，则返回
        推入的 (key, value) 本身。这里排序只根据 value 是否更大了判断，因此如果有的情况是越小越好，请在输入前取负号。

        :param str key:
        :param float value: 如果为 None， 则不做任何操作。
        :return: （1）返回输入的 (key, value) ，说明不满足 topk; (2) 返回(None, None)，说明满足 topk 且没有被挤出过去的记录; (3)
            返回非输入的 (key, value) , 说明输入满足 topk，且挤出了之前的记录。
        """
        if value is None:
            return key, value
        if self.topk < 0:
            return None, None
        if self.topk == 0:
            return key, value
        if len(self.topk_dict)<self.topk:
            self.topk_dict[key] = value
            return None, None
        min_key = min(self.topk_dict, key=lambda x:self.topk_dict[x])
        if self.topk_dict[min_key] > value:
            return key, value
        else:
            min_value = self.topk_dict.pop(min_key)
            self.topk_dict[key] = value
            return min_key, min_value

    def state_dict(self):
        return deepcopy(self.topk_dict)

    def load_state_dict(self, states):
        self.topk_dict.update(states)

    def __str__(self):
        return f'topk-{self.topk}'

    def __bool__(self):
        # 当 topk 为 0 时，表明该 topk_queue 无意义。
        return self.topk != 0


class TopkSaver(ResultsMonitor, Saver):
    """
    用来识别 topk 模型并保存，也可以仅当一个保存 Saver 使用。保存路径为::

        - folder/
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}-{topk_monitor}_{monitor_value}/  # 满足topk条件存储文件名

    :param topk: 保存表现最好的 ``topk`` 个模型，-1 为保存所有模型；0 为都不保存；大于 0 的数为保存 ``topk`` 个；
    :param monitor: 监控的 metric 值：

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: 该 monitor 是否越大越好。
    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param save_object: 可选 ``['trainer', 'model']`` ，表示在保存时的保存对象为 ``trainer+model`` 还是 只是 ``model`` 。如果
        保存 ``trainer`` 对象的话，将会保存 :class:`~fastNLP.core.controllers.Trainer` 的相关状态，可以通过 :meth:`Trainer.load_checkpoint` 加载该断
        点继续训练。如果保存的是 ``Model`` 对象，则可以通过 :meth:`Trainer.load_model` 加载该模型权重。
    :param only_state_dict: 保存时是否仅保存权重，在 ``model_save_fn`` 不为 None 时无意义。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函数应当接受一个文件夹作为参数，不返回任何东西。
        如果传入了 ``model_save_fn`` 函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运行该函数。
    :param save_evaluate_results: 是否保存 evaluate 的结果。如果为 True ，在保存 topk 模型的 folder 中还将额外保存一个
        ``fastnlp_evaluate_results.json`` 文件，记录当前的 metric results 。仅在设置了 ``topk`` 的场景下有用，默认为 True 。
    :param kwargs: 更多需要传递给 :meth:`Trainer.save_checkpoint` 或者 :meth:`Trainer.save_model` 接口的参数。
    """
    def __init__(self, topk:int=0, monitor:str=None, larger_better:bool=True, folder:str=None, save_object:str='model',
                 only_state_dict:bool=True, model_save_fn:Callable=None, save_evaluate_results:bool=True,
                 **kwargs):
        if topk is None:
            topk = 0
        ResultsMonitor.__init__(self, monitor, larger_better)
        Saver.__init__(self, folder, save_object, only_state_dict, model_save_fn, **kwargs)

        if monitor is not None and topk == 0:
            raise RuntimeError("`monitor` is set, but `topk` is 0.")
        if topk != 0 and monitor is None:
            raise RuntimeError("`topk` is set, but `monitor` is None.")

        self.topk_queue = TopkQueue(topk)
        self.save_evaluate_results = save_evaluate_results

    @rank_zero_call
    def save_topk(self, trainer, results: Dict) -> Optional[str]:
        """
        根据 ``results`` 是否满足 topk 的相关设定决定是否保存，如果发生了保存，将返回保存的文件夹。如果返回为 ``None`` ，则说明此次没有满足
        topk 要求，没有发生保存。

        :param trainer:
        :param results: evaluate 的结果。
        :return:
        """
        if self.monitor is not None and self.topk_queue:
            monitor_value = self.get_monitor_value(results)
            if monitor_value is None:
                return
            key = f"{self.save_object}-epoch_{trainer.cur_epoch_idx}-batch_{trainer.global_forward_batches}" \
                  f"-{self.monitor_name}_{monitor_value}"
            pop_key, pop_value = self.topk_queue.push(key, monitor_value if self.larger_better else -monitor_value)
            if pop_key == key:  # 说明不足以构成 topk，被退回了
                return None
            folder = self.save(trainer, key)
            if self.save_evaluate_results and folder:
                try:
                    self.save_json(self.itemize_results(results),
                                         os.path.join(folder, FASTNLP_EVALUATE_RESULT_FILENAME))
                except:
                    logger.exception(f"Fail to save evaluate results to {folder}")

            if pop_key and pop_key != key:  # 说明需要移除之前的 topk
                self.rm(pop_key)
            return folder

    def state_dict(self):
        states = {
            'topk_queue': self.topk_queue.state_dict(),
            'timestamp_path': str(self.timestamp_path),
        }
        if isinstance(self._real_monitor, str):
            states['_real_monitor'] = self._real_monitor

        return states

    def load_state_dict(self, states):
        topk_queue_states = states['topk_queue']
        self.topk_queue.load_state_dict(topk_queue_states)

        timestamp_path = states['timestamp_path']
        if not os.path.exists(timestamp_path):
            logger.info(f"The resuming checkpoint folder {timestamp_path} is not exists, checkpoint will save to "
                        f" {self.timestamp_path.absolute()}.")
        else:
            logger.info(f"Resume to save checkpoint in path: {timestamp_path}.")
            self.timestamp_path = Path(timestamp_path)

        if '_real_monitor' in states:
            self._real_monitor = states["_real_monitor"]

    def __str__(self):
        return f'topk-{self.topk_queue}#save_object-{self.save_object}'

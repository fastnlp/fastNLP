import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_LAUNCH_TIME, rank_zero_call
from fastNLP.envs.env import FASTNLP_EVALUATE_RESULT_FILENAME
from ...envs.distributed import rank_zero_rm
from .has_monitor_callback import ResultsMonitor

__all__ = ['TopkSaver']


class Saver:
    r"""
    执行保存操作的类，包含模型或断电的保存函数。

    保存的文件组织结构为::

        - folder  # 当前初始化的参数
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - folder_name  # 由 save() 调用时传入。

    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param save_object: 可选 ``['trainer', 'model']``，表示在保存时的保存对象为
        ``trainer+model`` 还是 只是 ``model``。如果保存 ``trainer`` 对象的话，将
        会保存 :class:`.Trainer` 的相关状态，之后可以通过 :meth:`.Trainer.\
        load_checkpoint` 加载该断点继续进行训练。如果保存的是 ``Model`` 对象，则可
        以通过 :meth:`.Trainer.load_model` 加载该模型权重。
    :param only_state_dict: 保存时是否仅保存权重，在 ``model_save_fn`` 不为
        ``None`` 时无意义。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函
        数应当接受一个文件夹作为参数，不返回任何东西。如果传入了 ``model_save_fn``
        函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运
        行该函数。
    :kwargs: 包含以下额外参数，以及更多需要传递给 :meth:`.Trainer.save_checkpoint`
        或者  :meth:`.Trainer.save_model` 接口的参数。

        * *use_timestamp_folder* (``bool``) -- 是否创建以脚本的启动时间命名的文件
          夹，默认为 ``True``。
    """

    def __init__(self,
                 folder: Optional[Union[str, Path]] = None,
                 save_object: str = 'model',
                 only_state_dict: bool = True,
                 model_save_fn: Optional[Callable] = None,
                 **kwargs):
        if folder is None:
            folder = Path.cwd().absolute()
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        elif folder.is_file():
            raise ValueError(
                'Parameter `folder` should be a directory instead of a file.')

        self.folder = folder
        self.only_state_dict = only_state_dict
        self.model_save_fn = model_save_fn
        self.kwargs = kwargs
        self.save_object = save_object
        self.save_fn_name = 'save_checkpoint' if save_object == 'trainer' else 'save_model'
        self.use_timestamp_folder = kwargs.get('use_timestamp_folder', True)

        self.save_folder = self.folder.joinpath(
            os.environ[FASTNLP_LAUNCH_TIME]
        ) if self.use_timestamp_folder else self.folder
        # 打印这次运行时 checkpoint 所保存在的文件夹，因为这个文件夹是根据时间
        # 实时生成的，因此需要打印出来防止用户混淆；
        logger.info('The checkpoint will be saved in this folder '
                    f'for this time: {self.save_folder}.')

    def save(self, trainer, folder_name):
        """
        执行保存的函数，将数据保存在::

            - folder/
                - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                    - folder_name  # 当前函数参数

        :param trainer: Trainer 对象
        :param folder_name: 保存的 folder 名称，将被创建。
        :return: 实际发生保存的 folder 绝对路径。如果为 None 则没有创建。
        """
        folder = self.save_folder.joinpath(folder_name)

        folder.mkdir(parents=True, exist_ok=True)

        save_fn = getattr(trainer, self.save_fn_name)
        save_fn(
            folder=folder,
            only_state_dict=self.only_state_dict,
            model_save_fn=self.model_save_fn,
            **self.kwargs)
        # TODO 如果 Metric 没有进行聚集操作，此时会创建出多个文件夹且只在 rank 0 的
        # 文件夹中进行保存
        # 可能的解决方法：检测出空文件夹并且删除

        return str(os.path.abspath(folder))

    @rank_zero_call
    def save_json(self, results, path):
        """以 json 格式保存 results 到 path 中。

        :param results: 一般是评测后的结果。
        :param path: 保存的文件名
        :return:
        """
        with open(path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2)

    @rank_zero_call
    def rm(self, folder_name):
        r"""移除 folder/timestamp/folder_name 。其中 folder 为用户在初始化指定，
        timestamp 为当前脚本的启动时间。

        :param folder_name: 需要移除的路径。
        :return:
        """
        folder = self.save_folder.joinpath(folder_name)
        rank_zero_rm(folder)

    def state_dict(self):
        states = {'save_folder': str(self.save_folder)}
        return states

    def load_state_dict(self, states):
        save_folder = states['save_folder']
        # 用户手动传入的 folder 应有最高的优先级
        if self.folder is not None:
            logger.info(
                'Detected: The checkpoint was previously saved in '
                f'{save_folder}, different from the folder {self.save_folder} '
                'you provided, what you provide has higher priority.')
        elif not os.path.exists(save_folder):
            logger.info(
                f'The resuming checkpoint folder {save_folder} is not exists, '
                f'checkpoint will save to  {self.save_folder.absolute()}.')
        else:
            logger.info(f'Resume to save checkpoint in path: {save_folder}.')
            self.save_folder = Path(save_folder)

    def __str__(self):
        return f'saver:{self.save_object}'


class TopkQueue:
    """用于维护处于 topk 的 key, value 对。

    :param int topk: 整数，-1 表示所有数据都是 topk 的; 如果是 0, 表示没有任何数据
        是满足 topk 的。
    """

    def __init__(self, topk):
        assert isinstance(topk, int)
        self.topk = topk
        self.topk_dict = {}  # 其中 key 为保存的内容，value 是对应的性能。

    def push(self, key, value) -> Tuple[Union[str, None], Union[float, None]]:
        r"""将 key/value 推入 topk 的 queue 中，以 value 为标准，如果满足 topk 则
        保留此次推入的信息，同时如果新推入的数据将之前的数据挤出了 topk ，则会返回被
        挤出的 (key, value)；如果返回为 (None, None)，说明满足 topk 且没有数据被挤
        出。如果不满足 topk ，则返回推入的 (key, value) 本身。这里排序只根据 value
        是否更大了判断，因此如果有的情况是越小越好，请在输入前取负号。

        :param str key:
        :param float value: 如果为 None，则不做任何操作。
        :return: （1）返回输入的 (key, value) ，说明不满足 topk;
            (2) 返回(None, None)，说明满足 topk 且没有被挤出过去的记录;
            (3)返回非输入的 (key, value) , 说明输入满足 topk，且挤出了之前的记录。
        """
        if value is None:
            return key, value
        if self.topk < 0:
            return None, None
        if self.topk == 0:
            return key, value
        if len(self.topk_dict) < self.topk:
            self.topk_dict[key] = value
            return None, None
        min_key = min(self.topk_dict, key=lambda x: self.topk_dict[x])
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
    r"""用于识别 topk 模型并保存，也可以仅当一个保存 Saver 使用。

    保存路径为::

        - folder/
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}-{topk_monitor}_{monitor_value}/  # 满足topk条件存储文件名

    :param topk: 保存表现最好的 ``topk`` 个模型，-1 为保存所有模型；0 为都不保
        存；大于 0 的数为保存 ``topk`` 个；
    :param monitor: 监控的 metric 值：

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`.Trainer` 中设置的 `monitor` 值（如果有设
          置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: 该 monitor 是否越大越好。
    :param folder: 保存在哪个文件夹下，默认为当前 folder 下。
    :param save_object: 可选 ``['trainer', 'model']``，表示在保存时的保存对象为
        ``trainer+model`` 还是 只是 ``model``。如果保存 ``trainer`` 对象的话，将
        会保存 :class:`.Trainer` 的相关状态，之后可以通过 :meth:`.Trainer.\
        load_checkpoint` 加载该断点继续进行训练。如果保存的是 ``Model`` 对象，则可
        以通过 :meth:`.Trainer.load_model` 加载该模型权重。
    :param only_state_dict: 保存时是否仅保存权重，在 ``model_save_fn`` 不为
        ``None`` 时无意义。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函
        数应当接受一个文件夹作为参数，不返回任何东西。如果传入了 ``model_save_fn``
        函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运
        行该函数。
    :kwargs: 包含以下额外参数，以及更多需要传递给 :meth:`.Trainer.save_checkpoint`
        或者  :meth:`.Trainer.save_model` 接口的参数。

        * *use_timestamp_folder* (``bool``) -- 是否创建以脚本的启动时间命名的文件
          夹，默认为 ``True``。
    """

    def __init__(self,
                 topk: int = 0,
                 monitor: Optional[Union[str, Callable]] = None,
                 larger_better: bool = True,
                 folder: Optional[Union[str, Path]] = None,
                 save_object: str = 'model',
                 only_state_dict: bool = True,
                 model_save_fn: Optional[Callable] = None,
                 save_evaluate_results: bool = True,
                 **kwargs):
        if topk is None:
            topk = 0
        ResultsMonitor.__init__(self, monitor, larger_better)
        Saver.__init__(self, folder, save_object, only_state_dict,
                       model_save_fn, **kwargs)

        if monitor is not None and topk == 0:
            raise RuntimeError('`monitor` is set, but `topk` is 0.')
        if topk != 0 and monitor is None:
            raise RuntimeError('`topk` is set, but `monitor` is None.')

        self.topk_queue = TopkQueue(topk)
        self.save_evaluate_results = save_evaluate_results

    # 注意这里我们为了支持 torch_fsdp 去除了 ''@rank_zero_call''；
    def save_topk(self, trainer, results: Dict) -> Optional[str]:
        r"""根据 ``results`` 是否满足 topk 的相关设定决定是否保存，如果发生了保存，
        将返回保存的文件夹。如果返回为 ``None``，则说明此次没有满足 topk 要求，没
        有发生保存。

        :param trainer:
        :param results: evaluate 的结果。
        :return: 如果满足 topk 的相关设定，则返回保存的文件夹；否则返回 ``None``。
        """
        if self.monitor is not None and self.topk_queue:
            monitor_value = self.get_monitor_value(results)
            if monitor_value is None:
                return None
            key = f'{self.save_object}-epoch_{trainer.cur_epoch_idx}-' \
                  f'batch_{trainer.global_forward_batches}' \
                  f'-{self.monitor_name}_{monitor_value}'
            pop_key, pop_value = self.topk_queue.push(
                key, monitor_value if self.larger_better else -monitor_value)
            if pop_key == key:  # 说明不足以构成 topk，被退回了
                return None
            folder = self.save(trainer, key)
            if self.save_evaluate_results and folder:
                try:
                    self.save_json(
                        self.itemize_results(results),
                        os.path.join(folder, FASTNLP_EVALUATE_RESULT_FILENAME))
                except Exception:
                    logger.exception(
                        f'Fail to save evaluate results to {folder}')

            if pop_key and pop_key != key:  # 说明需要移除之前的 topk
                self.rm(pop_key)
            return folder
        else:
            return None

    def state_dict(self):
        states = {
            'topk_queue': self.topk_queue.state_dict(),
            'save_folder': str(self.save_folder),
        }
        if isinstance(self._real_monitor, str):
            states['_real_monitor'] = self._real_monitor

        return states

    def load_state_dict(self, states):
        topk_queue_states = states['topk_queue']
        self.topk_queue.load_state_dict(topk_queue_states)

        save_folder = states['save_folder']
        # 用户手动传入的 folder 应有最高的优先级
        if self.folder is not None:
            logger.info(
                'Detected: The checkpoint was previously saved in '
                f'{save_folder}, different from the folder {self.save_folder} '
                'you provided, what you provide has higher priority.')
        elif not os.path.exists(save_folder):
            logger.info(
                f'The resuming checkpoint folder {save_folder} is not exists, '
                f'checkpoint will save to {self.save_folder.absolute()}.')
        else:
            logger.info(f'Resume to save checkpoint in path: {save_folder}.')
            self.save_folder = Path(save_folder)

        if '_real_monitor' in states:
            self._real_monitor = states['_real_monitor']

    def __str__(self):
        return f'topk-{self.topk_queue}#save_object-{self.save_object}'

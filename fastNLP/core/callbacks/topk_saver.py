import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Tuple

from fastNLP.core.utils import rank_zero_rm
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_LAUNCH_TIME
from fastNLP.envs import rank_zero_call
from fastNLP.envs.env import FASTNLP_EVALUATE_RESULT_FILENAME
from .has_monitor_callback import MonitorUtility


class Saver:
    def __init__(self, folder, only_state_dict, model_save_fn, **kwargs):
        """
        执行保存的对象。保存的文件组织结构为
        - folder  # 当前初始化的参数
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - folder_name  # 由 save() 调用时传入。

        :param folder:
        :param only_state_dict:
        :param model_save_fn:
        :param kwargs:
        """
        if folder is None:
            logger.warning(
                "Parameter `folder` is None, and we will use the current work directory to find and load your model.")
            folder = Path.cwd()
        folder = Path(folder)
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        elif folder.is_file():
            raise ValueError("Parameter `folder` should be a directory instead of a file.")

        self.folder = folder
        self.only_state_dict = only_state_dict
        self.model_save_fn = model_save_fn
        self.kwargs = kwargs
        self.eval_results = kwargs.get('eval_results', True)
        self.timestamp_path = self.folder.joinpath(os.environ[FASTNLP_LAUNCH_TIME])

    @rank_zero_call
    def save(self, save_fn, folder_name):
        """
        执行保存的函数，将数据保存在 folder/timestamp/folder_name 下。其中 folder 为用户在初始化指定，
            timestamp 为当前脚本的启动时间。

        :param save_fn: 调用的保存函数，应该可接受参数 folder:str, only_state_dict: bool, model_save_fn: callable, kwargs
        :param folder_name: 保存的 folder 名称，将被创建。
        :return: 返回实际发生保存的 folder 绝对路径。如果为 None 则没有创建。
        """
        folder = self.timestamp_path.joinpath(folder_name)
        folder.mkdir(parents=True, exist_ok=True)
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

        :param results:
        :param path:
        :return:
        """
        with open(path, 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2)

    @rank_zero_call
    def rm(self, folder_name):
        """
        移除 folder/timestamp/folder_name 。其中 folder 为用户在初始化指定, timestamp 为当前脚本的启动时间。

        :param folder_name:
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
        return 'saver'  # saver是无状态的，不需要有特定名字


class TopkQueue:
    def __init__(self, topk):
        """
        用于维护处于 topk 的 key, value 对。

        :param int topk: 整数，-1 表示所有数据都是 topk 的; 如果是 0, 表示没有任何数据是满足 topk 的。
        """
        assert isinstance(topk, int)
        self.topk = topk
        self.topk_dict = {}  # 其中 key 为保存的

    def push(self, key, value) -> Optional[Tuple[str, float]]:
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
        # 仅当 topk 为 0 时，表明该 topk_queue 无意义。
        return self.topk != 0


class TopkSaver(MonitorUtility, Saver):
    def __init__(self, topk, monitor, larger_better, folder, only_state_dict,
                 model_save_fn, save_evaluate_results,
                 save_object, **kwargs):
        """
        用来保存识别 tokp 模型并保存。

        :param topk:
        :param monitor:
        :param larger_better:
        :param folder:
        :param only_state_dict:
        :param model_save_fn:
        :param save_evaluate_results:
        :param save_object:
        :param kwargs:
        """
        MonitorUtility.__init__(self, monitor, larger_better)
        Saver.__init__(self, folder, only_state_dict, model_save_fn, **kwargs)

        if monitor is not None and topk == 0:
            raise RuntimeError("`monitor` is set, but `topk` is 0.")
        if topk != 0 and monitor is None:
            raise RuntimeError("`topk` is set, but `monitor` is None.")

        assert save_object in ['trainer', 'model']

        self.saver = Saver(folder, only_state_dict, model_save_fn, **kwargs)
        self.topk_queue = TopkQueue(topk)
        self.save_evaluate_results = save_evaluate_results
        self.save_object = save_object
        self.save_fn_name = 'save' if save_object == 'trainer' else 'save_model'

    @rank_zero_call
    def save_topk(self, trainer, results: Dict) -> Optional[str]:
        """
        根据 results 是否满足 topk 的相关设定决定是否保存，如果发生了保存，将返回保存的文件夹。

        :param trainer:
        :param results:
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

    def save(self, trainer, folder_name):
        fn = getattr(trainer, self.save_fn_name)
        return super().save(fn, folder_name)

    def state_dict(self):
        states = {
            'topk_queue': self.topk_queue.state_dict(),
            'saver': self.saver.state_dict()
        }
        if isinstance(self._real_monitor, str):
            states['_real_monitor'] = self._real_monitor

        return states

    def load_state_dict(self, states):
        topk_queue_states = states['topk_queue']
        saver_states = states['saver']
        self.topk_queue.load_state_dict(topk_queue_states)
        self.saver.load_state_dict(saver_states)
        if '_real_monitor' in states:
            self._real_monitor = states["_real_monitor"]

    def __str__(self):
        return f'topk-{self.topk_queue}#saver-{self.saver}#save_object-{self.save_object}'

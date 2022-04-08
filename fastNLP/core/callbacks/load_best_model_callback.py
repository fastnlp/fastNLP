__all__ = [
    'LoadBestModelCallback'
]

import os
from typing import Optional, Callable
from .callback import Callback
from .utils import _get_monitor_value
from io import BytesIO
import shutil

from fastNLP.envs.env import FASTNLP_LAUNCH_TIME, FASTNLP_GLOBAL_RANK, FASTNLP_BACKEND_LAUNCH
from fastNLP.core.log import logger
from fastNLP.envs import all_rank_call


class LoadBestModelCallback(Callback):
    def __init__(self, monitor:str, larger_better:bool = True, only_state_dict:bool = True,
                 save_folder:Optional[str] = None, model_save_fn:Optional[Callable] = None,
                 model_load_fn:Optional[Callable] = None,
                 delete_after_train:bool = True):
        """
        保存最佳的 monitor 值最佳的模型，并在训练结束的时候重新加载模型。仅在训练正常结束的时候才能加载最好的模型。

        :param str monitor: 监控的 metric 值。
        :param larger_better: 该 metric 值是否是越大越好。
        :param save_folder: 保存的文件夹，如果为空，则保存在内存中。不为空，则保存一份权重到文件中，当为多机训练，且本值不为空时，请确保
            不同的机器均可访问当该路径。当 model_save_fn 不为 None 时该值一定不能为空。
        :param only_state_dict: 是否只保存模型的参数。当 model_save_fn 不为空时，该值无效。
        :param model_save_fn: 保存 model 的函数，与 model_load_fn 必须同时不为空。本函数的输入为一个已经创建好的文件夹，没有输出，
            请在函数内完成对模型的保存。
        :param model_load_fn: 加载 model 的函数，与 model_save_fn 必须同时不为空。本函数的输入为一个已经创建好的文件夹，没有输出，
            请在函数内完成对模型的加载。
        :param delete_after_train: 在加载了最佳模型之后是否删掉模型。
        """
        if model_load_fn is not None:
            assert callable(model_load_fn), "`model_load_fn` must be a callable object."
            assert model_save_fn is not None, "`model_load_fn` and `model_save_fn` must be passed at the same time."
        if model_save_fn is not None:
            assert callable(model_save_fn), "`model_save_fn` must be a callable object."
            assert model_load_fn is not None, "`model_load_fn` and `model_save_fn` must be passed at the same time."

        if model_save_fn is not None:
            assert save_folder is not None, "When passing `model_save_fn`, `save_folder` must be provided."

        if save_folder is not None:
            if os.path.exists(save_folder):
                assert os.path.isdir(save_folder), f"`save_folder` must be a directory."
            else:
                os.makedirs(save_folder, exist_ok=True)
            save_folder = os.path.join(save_folder, os.environ.get(FASTNLP_LAUNCH_TIME))
            self.real_save_folder = os.path.join(save_folder, 'best_so_far')
            if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                os.makedirs(self.real_save_folder)
        else:  # 创建出一个 stringio
            self.real_save_folder = None
            self.buffer = BytesIO()

        self.monitor = monitor
        self.larger_better = larger_better
        self.save_folder = save_folder
        self.only_state_dict = only_state_dict
        self.model_save_fn = model_save_fn
        self.model_load_fn = model_load_fn
        self.delete_after_after = delete_after_train
        self._real_monitor = None
        self.monitor_value = float('-inf') if larger_better else float('inf')

    def on_after_trainer_initialized(self, trainer, driver):
        if self.save_folder is not None and driver.is_distributed() and int(os.environ.get(FASTNLP_BACKEND_LAUNCH, 0))==1:
            # 如果需要保存，但是又是不是 fastNLP 拉起的, 需要同步一下 folder
            try:
                self.real_save_folder = driver.broadcast_object(self.real_save_folder, src=0, group=None)
                logger.debug(f"Synchronize best model save folder: {self.real_save_folder} for LoadBestModelCallback.")
            except NotImplementedError:
                raise RuntimeError(f"Currently {driver.__class__.__name__} does not support using `save_folder` to "
                                   f"save best model when launch using script.")

    def on_validate_end(self, trainer, results):
        self._real_monitor, monitor_value = _get_monitor_value(monitor=self.monitor,
                                                               real_monitor=self._real_monitor,
                                                               res=results)
        if (monitor_value < self.monitor_value and self.larger_better is False) or \
            (monitor_value > self.monitor_value  and self.larger_better):
            self.monitor_value = monitor_value
            if self.real_save_folder:
                trainer.save_model(folder=self.real_save_folder, only_state_dict=self.only_state_dict,
                                   model_save_fn=self.model_save_fn)
            else:
                self.buffer.seek(0)
                with all_rank_call():
                    trainer.save_model(folder=self.buffer, only_state_dict=self.only_state_dict)

    def on_train_end(self, trainer):
        logger.info(f"Loading best model with {self._real_monitor}: {self.monitor_value}...")
        if self.real_save_folder:
            trainer.load_model(folder=self.real_save_folder, only_state_dict=self.only_state_dict,
                               model_load_fn=self.model_load_fn)
        else:
            self.buffer.seek(0)
            trainer.load_model(folder=self.buffer, only_state_dict=self.only_state_dict)

        if self.delete_after_after:
            if self.real_save_folder and int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                # 只需要 rank 0 执行删除。
                logger.info(f"Deleting {self.real_save_folder}...")
                shutil.rmtree(self.real_save_folder)
                try:
                    # 如果是 emtpy 的，就会被删除掉
                    os.rmdir(self.save_folder)
                except:
                    pass
            elif hasattr(self, 'buffer'):
                self.buffer.close()
                del self.buffer

    def on_exception(self, trainer, exception):
        if self.delete_after_after:
            if self.real_save_folder: # 这里，谁处异常，谁删除
                logger.info(f"Deleting {self.real_save_folder}...")
                shutil.rmtree(self.real_save_folder)
                try:
                    # 如果是 emtpy 的，就会被删除掉
                    os.rmdir(self.save_folder)
                except:
                    pass
            elif hasattr(self, 'buffer'):
                self.buffer.close()
                del self.buffer
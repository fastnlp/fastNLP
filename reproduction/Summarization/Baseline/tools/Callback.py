#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import sys
import time
import numpy as np

import torch

from fastNLP.core.const import Const
from fastNLP.io.model_io import ModelSaver
from fastNLP.core.callback import Callback, EarlyStopError

from fastNLP.core._logger import logger

class TrainCallback(Callback):
    def __init__(self, hps, patience=3, quit_all=True):
        super().__init__()
        self._hps = hps
        self.patience = patience
        self.wait = 0
        self.train_loss = 0.0
        self.prev_train_avg_loss = 1000.0
        self.train_dir = os.path.join(self._hps.save_root, "train")

        if type(quit_all) != bool:
            raise ValueError("In KeyBoardInterrupt, quit_all arguemnt must be a bool.")
        self.quit_all = quit_all

    def on_epoch_begin(self):
        self.epoch_start_time = time.time()
        self.model.Train = True

    def on_backward_begin(self, loss):
        """
        
        :param loss: []
        :return: 
        """
        if not (np.isfinite(loss.data)).numpy():
            logger.error("train Loss is not finite. Stopping.")
            logger.info(loss)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    logger.info(name)
                    logger.info(param.grad.data.sum())
            raise Exception("train Loss is not finite. Stopping.")
        self.train_loss += loss.data


    def on_backward_end(self):
        if self._hps.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hps.max_grad_norm)
        torch.cuda.empty_cache()

    def on_epoch_end(self):
        epoch_avg_loss = self.train_loss / self.n_steps
        logger.info('   | end of epoch {:3d} | time: {:5.2f}s | train loss: {:5.6f}'
                    .format(self.epoch, (time.time() - self.epoch_start_time), epoch_avg_loss))
        if self.prev_train_avg_loss < epoch_avg_loss:
            save_file = os.path.join(self.train_dir, "earlystop.pkl")
            self.save_model(save_file)
        else:
            self.prev_train_avg_loss = epoch_avg_loss
            self.train_loss = 0.0

        # save epoch
        save_file = os.path.join(self.train_dir, "epoch_%d.pkl" % self.epoch)
        self.save_model(save_file)



    def on_valid_begin(self):
        self.valid_start_time = time.time()
        self.model.Train = False

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        logger.info('   | end of valid {:3d} | time: {:5.2f}s | '
                    .format(self.epoch, (time.time() - self.valid_start_time)))

        # early stop
        if not is_better_eval:
            if self.wait == self.patience:
                train_dir = os.path.join(self._hps.save_root, "train")
                save_file = os.path.join(train_dir, "earlystop.pkl")
                self.save_model(save_file)
                raise EarlyStopError("Early stopping raised.")
            else:
                self.wait += 1
        else:
            self.wait = 0

        # lr descent
        if self._hps.lr_descent:
            new_lr = max(5e-6, self._hps.lr / (self.epoch + 1))
            for param_group in list(optimizer.param_groups):
                param_group['lr'] = new_lr
            logger.info("[INFO] The learning rate now is %f", new_lr)


    def on_exception(self, exception):
        if isinstance(exception, KeyboardInterrupt):
            logger.error("[Error] Caught keyboard interrupt on worker. Stopping supervisor...")
            save_file = os.path.join(self.train_dir, "earlystop.pkl")
            self.save_model(save_file)

            if self.quit_all is True:
                sys.exit(0)  # 直接退出程序
            else:
                pass
        else:
            raise exception  # 抛出陌生Error

    def save_model(self, save_file):
        saver = ModelSaver(save_file)
        saver.save_pytorch(self.model)
        logger.info('[INFO] Saving model to %s', save_file)








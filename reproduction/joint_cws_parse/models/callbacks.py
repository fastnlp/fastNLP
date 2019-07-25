
from fastNLP.core.callback import Callback
import torch
from torch import nn

class OptimizerCallback(Callback):
    def __init__(self, optimizer, scheduler, update_every=4):
        super().__init__()

        self._optimizer = optimizer
        self.scheduler = scheduler
        self._update_every = update_every

    def on_backward_end(self):
        if self.step % self._update_every==0:
            # nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 5)
            # self._optimizer.step()
            self.scheduler.step()
            # self.model.zero_grad()


class DevCallback(Callback):
    def __init__(self, tester, metric_key='u_f1'):
        super().__init__()
        self.tester = tester
        setattr(tester, 'verbose', 0)

        self.metric_key = metric_key

        self.record_best = False
        self.best_eval_value = 0
        self.best_eval_res = None

        self.best_dev_res = None # 存取dev的表现

    def on_valid_begin(self):
        eval_res = self.tester.test()
        metric_name = self.tester.metrics[0].__class__.__name__
        metric_value = eval_res[metric_name][self.metric_key]
        if metric_value>self.best_eval_value:
            self.best_eval_value = metric_value
            self.best_epoch = self.trainer.epoch
            self.record_best = True
            self.best_eval_res = eval_res
        self.test_eval_res = eval_res
        eval_str = "Epoch {}/{}. \n".format(self.trainer.epoch, self.n_epochs) + \
                   self.tester._format_eval_results(eval_res)
        self.pbar.write(eval_str)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if self.record_best:
            self.best_dev_res = eval_result
            self.record_best = False
        if is_better_eval:
            self.best_dev_res_on_dev = eval_result
            self.best_test_res_on_dev = self.test_eval_res
            self.dev_epoch = self.epoch

    def on_train_end(self):
        print("Got best test performance in epoch:{}\n Test: {}\n Dev:{}\n".format(self.best_epoch,
                                                            self.tester._format_eval_results(self.best_eval_res),
                                                            self.tester._format_eval_results(self.best_dev_res)))
        print("Got best dev performance in epoch:{}\n Test: {}\n Dev:{}\n".format(self.dev_epoch,
                                                            self.tester._format_eval_results(self.best_test_res_on_dev),
                                                            self.tester._format_eval_results(self.best_dev_res_on_dev)))
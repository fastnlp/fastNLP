import os
import torch
import sys
from torch import nn

from fastNLP.core.callback import Callback
from fastNLP.core.utils import _get_model_device

class MyCallback(Callback):
    def __init__(self, args):
        super(MyCallback, self).__init__()
        self.args = args
        self.real_step = 0

    def on_step_end(self):
        if self.step % self.update_every == 0 and self.step > 0:
            self.real_step += 1
            cur_lr = self.args.max_lr * 100 * min(self.real_step ** (-0.5), self.real_step * self.args.warmup_steps**(-1.5))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = cur_lr

            if self.real_step % 1000 == 0:
                self.pbar.write('Current learning rate is {:.8f}, real_step: {}'.format(cur_lr, self.real_step))
    
    def on_epoch_end(self):
        self.pbar.write('Epoch {} is done !!!'.format(self.epoch))

def _save_model(model, model_name, save_dir, only_param=False):
    """ 存储不含有显卡信息的 state_dict 或 model
    :param model:
    :param model_name:
    :param save_dir: 保存的 directory
    :param only_param:
    :return:
    """
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, nn.DataParallel):
        model = model.module
    if only_param:
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, model_path)
    else:
        _model_device = _get_model_device(model)
        model.cpu()
        torch.save(model, model_path)
        model.to(_model_device)

class SaveModelCallback(Callback):
    """
    由于Trainer在训练过程中只会保存最佳的模型， 该 callback 可实现多种方式的结果存储。
    会根据训练开始的时间戳在 save_dir 下建立文件夹，在再文件夹下存放多个模型
    -save_dir
        -2019-07-03-15-06-36
            -epoch0step20{metric_key}{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
            -epoch1step40
        -2019-07-03-15-10-00
            -epoch:0step:20{metric_key}:{evaluate_performance}.pt   # metric是给定的metric_key, evaluate_perfomance是性能
    :param str save_dir: 将模型存放在哪个目录下，会在该目录下创建以时间戳命名的目录，并存放模型
    :param int top: 保存dev表现top多少模型。-1为保存所有模型
    :param bool only_param: 是否只保存模型权重
    :param save_on_exception: 发生exception时，是否保存一份当时的模型
    """
    def __init__(self, save_dir, top=5, only_param=False, save_on_exception=False):
        super().__init__()

        if not os.path.isdir(save_dir):
            raise IsADirectoryError("{} is not a directory.".format(save_dir))
        self.save_dir = save_dir
        if top < 0:
            self.top = sys.maxsize
        else:
            self.top = top
        self._ordered_save_models = []  # List[Tuple], Tuple[0]是metric， Tuple[1]是path。metric是依次变好的，所以从头删

        self.only_param = only_param
        self.save_on_exception = save_on_exception

    def on_train_begin(self):
        self.save_dir = os.path.join(self.save_dir, self.trainer.start_time)

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        metric_value = list(eval_result.values())[0][metric_key]
        self._save_this_model(metric_value)

    def _insert_into_ordered_save_models(self, pair):
        # pair:(metric_value, model_name)
        # 返回save的模型pair与删除的模型pair. pair中第一个元素是metric的值，第二个元素是模型的名称
        index = -1
        for _pair in self._ordered_save_models:
            if _pair[0]>=pair[0] and self.trainer.increase_better:
                break
            if not self.trainer.increase_better and _pair[0]<=pair[0]:
                break
            index += 1
        save_pair = None
        if len(self._ordered_save_models)<self.top or (len(self._ordered_save_models)>=self.top and index!=-1):
            save_pair = pair
            self._ordered_save_models.insert(index+1, pair)
        delete_pair = None
        if len(self._ordered_save_models)>self.top:
            delete_pair = self._ordered_save_models.pop(0)
        return save_pair, delete_pair

    def _save_this_model(self, metric_value):
        name = "epoch:{}_step:{}_{}:{:.6f}.pt".format(self.epoch, self.step, self.trainer.metric_key, metric_value)
        save_pair, delete_pair = self._insert_into_ordered_save_models((metric_value, name))
        if save_pair:
            try:
                _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)
            except Exception as e:
                print(f"The following exception:{e} happens when saves model to {self.save_dir}.")
        if delete_pair:
            try:
                delete_model_path = os.path.join(self.save_dir, delete_pair[1])
                if os.path.exists(delete_model_path):
                    os.remove(delete_model_path)
            except Exception as e:
                print(f"Fail to delete model {name} at {self.save_dir} caused by exception:{e}.")

    def on_exception(self, exception):
        if self.save_on_exception:
            name = "epoch:{}_step:{}_Exception:{}.pt".format(self.epoch, self.step, exception.__class__.__name__)
            _save_model(self.model, model_name=name, save_dir=self.save_dir, only_param=self.only_param)



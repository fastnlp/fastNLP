# Code Modified from https://github.com/carpedm20/ENAS-pytorch
import math
import numpy as np
import time
import torch

from datetime import datetime, timedelta

from torch.optim import Adam

try:
    from tqdm.auto import tqdm
except:
    from ..core.utils import _pseudo_tqdm as tqdm

from ..core.trainer import Trainer
from ..core.batch import Batch
from ..core.callback import CallbackManager, CallbackException
from ..core.dataset import DataSet
from ..core.utils import _move_dict_value_to_device
from . import enas_utils as utils
from ..core.utils import _build_args


def _get_no_grad_ctx_mgr():
    """Returns a the `torch.no_grad` context manager for PyTorch version >=
    0.4, or a no-op context manager otherwise.
    """
    return torch.no_grad()


class ENASTrainer(Trainer):
    """A class to wrap training code."""
    
    def __init__(self, train_data, model, controller, **kwargs):
        """Constructor for training algorithm.
        :param DataSet train_data: the training data
        :param torch.nn.modules.module model: a PyTorch model
        :param torch.nn.modules.module controller: a PyTorch model
        """
        self.final_epochs = kwargs['final_epochs']
        kwargs.pop('final_epochs')
        super(ENASTrainer, self).__init__(train_data, model, **kwargs)
        self.controller_step = 0
        self.shared_step = 0
        self.max_length = 35
        
        self.shared = model
        self.controller = controller
        
        self.shared_optim = Adam(
            self.shared.parameters(),
            lr=20.0,
            weight_decay=1e-7)
        
        self.controller_optim = Adam(
            self.controller.parameters(),
            lr=3.5e-4)
    
    def train(self, load_best_model=True):
        """
        :param bool load_best_model: 该参数只有在初始化提供了dev_data的情况下有效，如果True, trainer将在返回之前重新加载dev表现
            最好的模型参数。
        :return results: 返回一个字典类型的数据,
            内含以下内容::

                seconds: float, 表示训练时长
                以下三个内容只有在提供了dev_data的情况下会有。
                best_eval: Dict of Dict, 表示evaluation的结果
                best_epoch: int，在第几个epoch取得的最佳值
                best_step: int, 在第几个step(batch)更新取得的最佳值

        """
        results = {}
        if self.n_epochs <= 0:
            print(f"training epoch is {self.n_epochs}, nothing was done.")
            results['seconds'] = 0.
            return results
        try:
            if torch.cuda.is_available() and "cuda" in self.device:
                self.model = self.model.cuda()
            self._model_device = self.model.parameters().__next__().device
            self._mode(self.model, is_test=False)
            
            self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            start_time = time.time()
            print("training epochs started " + self.start_time, flush=True)
            
            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()
            except (CallbackException, KeyboardInterrupt) as e:
                self.callback_manager.on_exception(e)
            
            if self.dev_data is not None:
                print(
                    "\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step) +
                    self.tester._format_eval_results(self.best_dev_perf), )
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step
                if load_best_model:
                    model_name = "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time])
                    load_succeed = self._load_model(self.model, model_name)
                    if load_succeed:
                        print("Reloaded the best model.")
                    else:
                        print("Fail to reload best model.")
        finally:
            pass
        results['seconds'] = round(time.time() - start_time, 2)
        
        return results
    
    def _train(self):
        if not self.use_tqdm:
            from fastNLP.core.utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm
        self.step = 0
        start = time.time()
        total_steps = (len(self.train_data) // self.batch_size + int(
            len(self.train_data) % self.batch_size != 0)) * self.n_epochs
        with inner_tqdm(total=total_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True) as pbar:
            avg_loss = 0
            data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=self.sampler, as_numpy=False,
                                  prefetch=self.prefetch)
            for epoch in range(1, self.n_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
                last_stage = (epoch > self.n_epochs + 1 - self.final_epochs)
                if epoch == self.n_epochs + 1 - self.final_epochs:
                    print('Entering the final stage. (Only train the selected structure)')
                # early stopping
                self.callback_manager.on_epoch_begin()
                
                # 1. Training the shared parameters omega of the child models
                self.train_shared(pbar)
                
                # 2. Training the controller parameters theta
                if not last_stage:
                    self.train_controller()
                
                if ((self.validate_every > 0 and self.step % self.validate_every == 0) or
                    (self.validate_every < 0 and self.step % len(data_iterator) == 0)) \
                        and self.dev_data is not None:
                    if not last_stage:
                        self.derive()
                    eval_res = self._do_validation(epoch=epoch, step=self.step)
                    eval_str = "Evaluation at Epoch {}/{}. Step:{}/{}. ".format(epoch, self.n_epochs, self.step,
                                                                                total_steps) + \
                               self.tester._format_eval_results(eval_res)
                    pbar.write(eval_str)
                
                # lr decay; early stopping
                self.callback_manager.on_epoch_end()
            # =============== epochs end =================== #
            pbar.close()
        # ============ tqdm end ============== #
    
    def get_loss(self, inputs, targets, hidden, dags):
        """Computes the loss for the same batch for M models.

        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.
        """
        if not isinstance(dags, list):
            dags = [dags]
        
        loss = 0
        for dag in dags:
            self.shared.setDAG(dag)
            inputs = _build_args(self.shared.forward, **inputs)
            inputs['hidden'] = hidden
            result = self.shared(**inputs)
            output, hidden, extra_out = result['pred'], result['hidden'], result['extra_out']
            
            self.callback_manager.on_loss_begin(targets, result)
            sample_loss = self._compute_loss(result, targets)
            loss += sample_loss
        
        assert len(dags) == 1, 'there are multiple `hidden` for multple `dags`'
        return loss, hidden, extra_out
    
    def train_shared(self, pbar=None, max_step=None, dag=None):
        """Train the language model for 400 steps of minibatches of 64
        examples.

        Args:
            max_step: Used to run extra training steps as a warm-up.
            dag: If not None, is used instead of calling sample().

        BPTT is truncated at 35 timesteps.

        For each weight update, gradients are estimated by sampling M models
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        model = self.shared
        model.train()
        self.controller.eval()
        
        hidden = self.shared.init_hidden(self.batch_size)
        
        abs_max_grad = 0
        abs_max_hidden_norm = 0
        step = 0
        raw_total_loss = 0
        total_loss = 0
        train_idx = 0
        avg_loss = 0
        data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=self.sampler, as_numpy=False,
                              prefetch=self.prefetch)
        
        for batch_x, batch_y in data_iterator:
            _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
            indices = data_iterator.get_batch_indices()
            # negative sampling; replace unknown; re-weight batch_y
            self.callback_manager.on_batch_begin(batch_x, batch_y, indices)
            # prediction = self._data_forward(self.model, batch_x)
            
            dags = self.controller.sample(1)
            inputs, targets = batch_x, batch_y
            # self.callback_manager.on_loss_begin(batch_y, prediction)
            loss, hidden, extra_out = self.get_loss(inputs,
                                                    targets,
                                                    hidden,
                                                    dags)
            hidden.detach_()
            
            avg_loss += loss.item()
            
            # Is loss NaN or inf? requires_grad = False
            self.callback_manager.on_backward_begin(loss)
            self._grad_backward(loss)
            self.callback_manager.on_backward_end()
            
            self._update()
            self.callback_manager.on_step_end()
            
            if (self.step + 1) % self.print_every == 0:
                if self.use_tqdm:
                    print_output = "loss:{0:<6.5f}".format(avg_loss / self.print_every)
                    pbar.update(self.print_every)
                else:
                    end = time.time()
                    diff = timedelta(seconds=round(end - start))
                    print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                        epoch, self.step, avg_loss, diff)
                pbar.set_postfix_str(print_output)
                avg_loss = 0
            self.step += 1
            step += 1
            self.shared_step += 1
            self.callback_manager.on_batch_end()
        # ================= mini-batch end ==================== #
    
    def get_reward(self, dag, entropies, hidden, valid_idx=0):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        
        data_iterator = Batch(self.dev_data, batch_size=self.batch_size, sampler=self.sampler, as_numpy=False,
                              prefetch=self.prefetch)
        
        for inputs, targets in data_iterator:
            valid_loss, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
            valid_loss = utils.to_item(valid_loss.data)
            
            valid_ppl = math.exp(valid_loss)
            
            R = 80 / valid_ppl
            
            rewards = R + 1e-4 * entropies
            
            return rewards, hidden
    
    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller
        model.train()
        # Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()
        
        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []
        
        hidden = self.shared.init_hidden(self.batch_size)
        total_loss = 0
        valid_idx = 0
        for step in range(20):
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)
            
            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                rewards, hidden = self.get_reward(dags,
                                                  np_entropies,
                                                  hidden,
                                                  valid_idx)
            
            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)
            
            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = 0.95
                baseline = decay * baseline + (1 - decay) * rewards
            
            adv = rewards - baseline
            adv_history.extend(adv)
            
            # policy loss
            loss = -log_probs * utils.get_variable(adv,
                                                   'cuda' in self.device,
                                                   requires_grad=False)
            
            loss = loss.sum()  # or loss.mean()
            
            # update
            self.controller_optim.zero_grad()
            loss.backward()
            
            self.controller_optim.step()
            
            total_loss += utils.to_item(loss.data)
            
            if ((step % 50) == 0) and (step > 0):
                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0
            
            self.controller_step += 1
            # prev_valid_idx = valid_idx
            # valid_idx = ((valid_idx + self.max_length) %
            #              (self.valid_data.size(0) - 1))
            # # Whenever we wrap around to the beginning of the
            # # validation data, we reset the hidden states.
            # if prev_valid_idx > valid_idx:
            #     hidden = self.shared.init_hidden(self.batch_size)
    
    def derive(self, sample_num=10, valid_idx=0):
        """We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        hidden = self.shared.init_hidden(self.batch_size)
        
        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)
        
        max_R = 0
        best_dag = None
        for dag in dags:
            R, _ = self.get_reward(dag, entropies, hidden, valid_idx)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag
        
        self.model.setDAG(best_dag)

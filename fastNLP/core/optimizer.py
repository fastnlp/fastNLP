import torch
import math
import numpy as np

class Optimizer(object):
    """

        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
        :param kwargs: additional parameters.
    """
    def __init__(self, model_params, **kwargs):
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs


class SGD(Optimizer):
    """

        :param float lr: learning rate. Default: 0.01
        :param float momentum: momentum. Default: 0
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    """

    def __init__(self, lr=0.001, momentum=0, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(model_params, **self.settings)
        else:
            return torch.optim.SGD(self.model_params, **self.settings)


class Adam(Optimizer):
    """

        :param float lr: learning rate
        :param float weight_decay:
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    """

    def __init__(self, lr=0.001, weight_decay=0, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adam, self).__init__(model_params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad,
                                   weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adam(model_params, **self.settings)
        else:
            return torch.optim.Adam(self.model_params, **self.settings)
 


class _YFOptimizer(object):
    def __init__(self, var_list, lr=0.1, mu=0.0, eps=1e-15, clip_thresh=None, weight_decay=0.0,
                 beta=0.999, curv_win_width=20, zero_debias=True, sparsity_debias=True, delta_mu=0.0,
                 auto_clip_fac=None, force_non_inc_step=False):
        '''
        clip thresh is the threshold value on ||lr * gradient||
        delta_mu can be place holder/variable/python scalar. They are used for additional
        momentum in situations such as asynchronous-parallel training. The default is 0.0
        for basic usage of the optimizer.
        Args:
          lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
          mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
          clip_thresh: python scalar. The manaully-set clipping threshold for tf.clip_by_global_norm.
            if None, the automatic clipping can be carried out. The automatic clipping
            feature is parameterized by argument auto_clip_fac. The auto clip feature
            can be switched off with auto_clip_fac = None
          beta: python scalar. The smoothing parameter for estimations.
          sparsity_debias: gradient norm and curvature are biased to larger values when
          calculated with sparse gradient. This is useful when the model is very sparse,
          e.g. LSTM with word embedding. For non-sparse CNN, turning it off could slightly
          accelerate the speed.
          delta_mu: for extensions. Not necessary in the basic use.
          force_non_inc_step: in some very rare cases, it is necessary to force ||lr * gradient||
          to be not increasing dramatically for stableness after some iterations.
          In practice, if turned on, we enforce lr * sqrt(smoothed ||grad||^2)
          to be less than 2x of the minimal value of historical value on smoothed || lr * grad ||.
          This feature is turned off by default.
        Other features:
          If you want to manually control the learning rates, self.lr_factor is
          an interface to the outside, it is an multiplier for the internal learning rate
          in YellowFin. It is helpful when you want to do additional hand tuning
          or some decaying scheme to the tuned learning rate in YellowFin.
          Example on using lr_factor can be found here:
          https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L109
        '''
        self._lr = lr
        self._mu = mu
        self._eps = eps
        # we convert var_list from generator to list so that
        # it can be used for multiple times
        self._var_list = list(var_list)
        self._clip_thresh = clip_thresh
        self._auto_clip_fac = auto_clip_fac
        self._beta = beta
        self._curv_win_width = curv_win_width
        self._zero_debias = zero_debias
        self._sparsity_debias = sparsity_debias
        self._force_non_inc_step = force_non_inc_step
        self._optimizer = torch.optim.SGD(self._var_list, lr=self._lr,
                                          momentum=self._mu, weight_decay=weight_decay)
        self._iter = 0
        # global states are the statistics
        self._global_state = {}

        # for decaying learning rate and etc.
        self._lr_factor = 1.0

    def state_dict(self):
        # for checkpoint saving
        sgd_state_dict = self._optimizer.state_dict()
        global_state = self._global_state
        lr_factor = self._lr_factor
        iter = self._iter
        lr = self._lr
        mu = self._mu
        clip_thresh = self._clip_thresh
        beta = self._beta
        curv_win_width = self._curv_win_width
        zero_debias = self._zero_debias
        h_min = self._h_min
        h_max = self._h_max

        return {
            "sgd_state_dict": sgd_state_dict,
            "global_state": global_state,
            "lr_factor": lr_factor,
            "iter": iter,
            "lr": lr,
            "mu": mu,
            "clip_thresh": clip_thresh,
            "beta": beta,
            "curv_win_width": curv_win_width,
            "zero_debias": zero_debias,
            "h_min": h_min,
            "h_max": h_max
        }

    def load_state_dict(self, state_dict):
        # for checkpoint saving
        self._optimizer.load_state_dict(state_dict['sgd_state_dict'])
        self._global_state = state_dict['global_state']
        self._lr_factor = state_dict['lr_factor']
        self._iter = state_dict['iter']
        self._lr = state_dict['lr']
        self._mu = state_dict['mu']
        self._clip_thresh = state_dict['clip_thresh']
        self._beta = state_dict['beta']
        self._curv_win_width = state_dict['curv_win_width']
        self._zero_debias = state_dict['zero_debias']
        self._h_min = state_dict["h_min"]
        self._h_max = state_dict["h_max"]
        return

    def set_lr_factor(self, factor):
        self._lr_factor = factor
        return

    def get_lr_factor(self):
        return self._lr_factor

    def zero_grad(self):
        self._optimizer.zero_grad()
        return

    def zero_debias_factor(self):
        return 1.0 - self._beta ** (self._iter + 1)

    def zero_debias_factor_delay(self, delay):
        # for exponentially averaged stat which starts at non-zero iter
        return 1.0 - self._beta ** (self._iter - delay + 1)

    def curvature_range(self):
        global_state = self._global_state
        if self._iter == 0:
            global_state["curv_win"] = torch.FloatTensor(self._curv_win_width, 1).zero_()
        curv_win = global_state["curv_win"]
        grad_norm_squared = self._global_state["grad_norm_squared"]
        curv_win[self._iter % self._curv_win_width] = np.log(grad_norm_squared + self._eps)
        valid_end = min(self._curv_win_width, self._iter + 1)
        # we use running average over log scale, accelerating
        # h_max / min in the begining to follow the varying trend of curvature.
        beta = self._beta
        if self._iter == 0:
            global_state["h_min_avg"] = 0.0
            global_state["h_max_avg"] = 0.0
            self._h_min = 0.0
            self._h_max = 0.0
        global_state["h_min_avg"] = \
            global_state["h_min_avg"] * beta + (1 - beta) * torch.min(curv_win[:valid_end])
        global_state["h_max_avg"] = \
            global_state["h_max_avg"] * beta + (1 - beta) * torch.max(curv_win[:valid_end])
        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
            self._h_min = np.exp(global_state["h_min_avg"] / debias_factor)
            self._h_max = np.exp(global_state["h_max_avg"] / debias_factor)
        else:
            self._h_min = np.exp(global_state["h_min_avg"])
            self._h_max = np.exp(global_state["h_max_avg"])
        if self._sparsity_debias:
            self._h_min *= self._sparsity_avg
            self._h_max *= self._sparsity_avg
        return

    def grad_variance(self):
        global_state = self._global_state
        beta = self._beta
        self._grad_var = np.array(0.0, dtype=np.float32)
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self._optimizer.state[p]

                if self._iter == 0:
                    state["grad_avg"] = grad.new().resize_as_(grad).zero_()
                    state["grad_avg_squared"] = grad.new().resize_as_(grad).zero_()
                state["grad_avg"].mul_(beta).add_(1 - beta, grad)
                self._grad_var += (torch.sum(state["grad_avg"] * state["grad_avg"]).numpy())
        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
        else:
            debias_factor = 1.0

        self._grad_var /= -(debias_factor ** 2)
        self._grad_var += (global_state['grad_norm_squared_avg'] / debias_factor).numpy()
        # in case of negative variance: the two term are using different debias factors
        self._grad_var = max(self._grad_var, self._eps)
        if self._sparsity_debias:
            self._grad_var *= self._sparsity_avg
        return

    def dist_to_opt(self):
        global_state = self._global_state
        beta = self._beta
        if self._iter == 0:
            global_state["grad_norm_avg"] = 0.0
            global_state["dist_to_opt_avg"] = 0.0
        global_state["grad_norm_avg"] = \
            global_state["grad_norm_avg"] * beta + (1 - beta) * math.sqrt(global_state["grad_norm_squared"])
        global_state["dist_to_opt_avg"] = \
            global_state["dist_to_opt_avg"] * beta \
            + (1 - beta) * global_state["grad_norm_avg"] / (global_state['grad_norm_squared_avg'] + self._eps)
        if self._zero_debias:
            debias_factor = self.zero_debias_factor()
            self._dist_to_opt = global_state["dist_to_opt_avg"] / debias_factor
        else:
            self._dist_to_opt = global_state["dist_to_opt_avg"]
        if self._sparsity_debias:
            self._dist_to_opt /= (np.sqrt(self._sparsity_avg) + self._eps)
        return

    def grad_sparsity(self):
        global_state = self._global_state
        if self._iter == 0:
            global_state["sparsity_avg"] = 0.0
        non_zero_cnt = 0.0
        all_entry_cnt = 0.0
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_non_zero = grad.nonzero()
                if grad_non_zero.dim() > 0:
                    non_zero_cnt += grad_non_zero.size()[0]
                all_entry_cnt += torch.numel(grad)
        beta = self._beta
        global_state["sparsity_avg"] = beta * global_state["sparsity_avg"] \
                                       + (1 - beta) * non_zero_cnt / float(all_entry_cnt)
        self._sparsity_avg = \
            global_state["sparsity_avg"] / self.zero_debias_factor()
        return

    def lr_grad_norm_avg(self):
        # this is for enforcing lr * grad_norm not
        # increasing dramatically in case of instability.
        #  Not necessary for basic use.
        global_state = self._global_state
        beta = self._beta
        if "lr_grad_norm_avg" not in global_state:
            global_state['grad_norm_squared_avg_log'] = 0.0
        global_state['grad_norm_squared_avg_log'] = \
            global_state['grad_norm_squared_avg_log'] * beta \
            + (1 - beta) * np.log(global_state['grad_norm_squared'] + self._eps)
        if "lr_grad_norm_avg" not in global_state:
            global_state["lr_grad_norm_avg"] = \
                0.0 * beta + (1 - beta) * np.log(self._lr * np.sqrt(global_state['grad_norm_squared']) + self._eps)
            # we monitor the minimal smoothed ||lr * grad||
            global_state["lr_grad_norm_avg_min"] = \
                np.exp(global_state["lr_grad_norm_avg"] / self.zero_debias_factor())
        else:
            global_state["lr_grad_norm_avg"] = global_state["lr_grad_norm_avg"] * beta \
                                               + (1 - beta) * np.log(
                self._lr * np.sqrt(global_state['grad_norm_squared']) + self._eps)
            global_state["lr_grad_norm_avg_min"] = \
                min(global_state["lr_grad_norm_avg_min"],
                    np.exp(global_state["lr_grad_norm_avg"] / self.zero_debias_factor()))

    def after_apply(self):
        # compute running average of gradient and norm of gradient
        beta = self._beta
        global_state = self._global_state
        if self._iter == 0:
            global_state["grad_norm_squared_avg"] = 0.0

        global_state["grad_norm_squared"] = 0.0
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                global_state['grad_norm_squared'] += torch.sum(grad * grad)

        global_state['grad_norm_squared_avg'] = \
            global_state['grad_norm_squared_avg'] * beta + (1 - beta) * global_state['grad_norm_squared']

        if self._sparsity_debias:
            self.grad_sparsity()

        self.curvature_range()
        self.grad_variance()
        self.dist_to_opt()

        if self._iter > 0:
            self.get_mu()
            self.get_lr()

            self._lr = beta * self._lr + (1 - beta) * self._lr_t
            self._mu = beta * self._mu + (1 - beta) * self._mu_t
        return

    def get_lr(self):
        self._lr_t = (1.0 - math.sqrt(self._mu_t)) ** 2 / (self._h_min + self._eps)
        return

    def get_cubic_root(self):
        # We have the equation x^2 D^2 + (1-x)^4 * C / h_min^2
        # where x = sqrt(mu).
        # We substitute x, which is sqrt(mu), with x = y + 1.
        # It gives y^3 + py = q
        # where p = (D^2 h_min^2)/(2*C) and q = -p.
        # We use the Vieta's substution to compute the root.
        # There is only one real solution y (which is in [0, 1] ).
        # http://mathworld.wolfram.com/VietasSubstitution.html
        # self._eps in the numerator is to prevent momentum = 1 in case of zero gradient
        p = (self._dist_to_opt + self._eps) ** 2 * (self._h_min + self._eps) ** 2 / 2 / (self._grad_var + self._eps)
        w3 = (-math.sqrt(p ** 2 + 4.0 / 27.0 * p ** 3) - p) / 2.0
        w = math.copysign(1.0, w3) * math.pow(math.fabs(w3), 1.0 / 3.0)
        y = w - p / 3.0 / (w + self._eps)
        x = y + 1
        return x

    def get_mu(self):
        root = self.get_cubic_root()
        dr = self._h_max / self._h_min
        self._mu_t = max(root ** 2, ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2)
        return

    def update_hyper_param(self):
        for group in self._optimizer.param_groups:
            group['momentum'] = self._mu
            if self._force_non_inc_step == False:
                group['lr'] = self._lr * self._lr_factor
            elif self._iter > self._curv_win_width:
                # force to guarantee lr * grad_norm not increasing dramatically.
                # Not necessary for basic use. Please refer to the comments
                # in YFOptimizer.__init__ for more details
                self.lr_grad_norm_avg()
                debias_factor = self.zero_debias_factor()
                group['lr'] = min(self._lr * self._lr_factor,
                                  2.0 * self._global_state["lr_grad_norm_avg_min"] \
                                  / np.sqrt(np.exp(self._global_state['grad_norm_squared_avg_log'] / debias_factor)))
        return

    def auto_clip_thresh(self):
        # Heuristic to automatically prevent sudden exploding gradient
        # Not necessary for basic use.
        return math.sqrt(self._h_max) * self._auto_clip_fac

    def step(self):
        # add weight decay
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

        if self._clip_thresh != None:
            torch.nn.utils.clip_grad_norm(self._var_list, self._clip_thresh)
        elif (self._iter != 0 and self._auto_clip_fac != None):
            # do not clip the first iteration
            torch.nn.utils.clip_grad_norm(self._var_list, self.auto_clip_thresh())

        # apply update
        self._optimizer.step()

        # after appply
        self.after_apply()

        # update learning rate and momentum
        self.update_hyper_param()

        self._iter += 1
        return

class YFOptimizer(Optimizer):
    """

        :param float lr: learning rate
        :param float weight_decay:
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
    """

    def __init__(self, lr=0.001, weight_decay=0, eps=1e-15, model_params=None):
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(YFOptimizer, self).__init__(model_params, lr=lr, eps=eps, weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return _YFOptimizer(model_params, **self.settings)
        else:
            return _YFOptimizer(self.model_params, **self.settings)

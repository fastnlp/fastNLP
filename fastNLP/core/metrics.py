import inspect
from collections import defaultdict

import numpy as np
import torch

from fastNLP.core.utils import CheckError
from fastNLP.core.utils import CheckRes
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_arg_dict_list
from fastNLP.core.utils import get_func_signature
from fastNLP.core.utils import seq_lens_to_masks


class MetricBase(object):
    def __init__(self):
        self.param_map = {}  # key is param in function, value is input param.
        self._checked = False

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def _init_param_map(self, key_map=None, **kwargs):
        """Check the validity of key_map and other param map. Add these into self.param_map

        :param key_map: dict
        :param kwargs:
        :return: None
        """
        value_counter = defaultdict(set)
        if key_map is not None:
            if not isinstance(key_map, dict):
                raise TypeError("key_map must be `dict`, got {}.".format(type(key_map)))
            for key, value in key_map.items():
                if value is None:
                    self.param_map[key] = key
                    continue
                if not isinstance(key, str):
                    raise TypeError(f"key in key_map must be `str`, not `{type(key)}`.")
                if not isinstance(value, str):
                    raise TypeError(f"value in key_map must be `str`, not `{type(value)}`.")
                self.param_map[key] = value
                value_counter[value].add(key)
        for key, value in kwargs.items():
            if value is None:
                self.param_map[key] = key
                continue
            if not isinstance(value, str):
                raise TypeError(f"in {key}={value}, value must be `str`, not `{type(value)}`.")
            self.param_map[key] = value
            value_counter[value].add(key)
        for value, key_set in value_counter.items():
            if len(key_set) > 1:
                raise ValueError(f"Several parameters:{key_set} are provided with one output {value}.")

        # check consistence between signature and param_map
        func_spect = inspect.getfullargspec(self.evaluate)
        func_args = [arg for arg in func_spect.args if arg != 'self']
        for func_param, input_param in self.param_map.items():
            if func_param not in func_args:
                raise NameError(
                    f"Parameter `{func_param}` is not in {get_func_signature(self.evaluate)}. Please check the "
                    f"initialization parameters, or change its signature.")

        # evaluate should not have varargs.
        if func_spect.varargs:
            raise NameError(f"Delete `*{func_spect.varargs}` in {get_func_signature(self.evaluate)}(Do not use "
                            f"positional argument.).")

    def get_metric(self, reset=True):
        raise NotImplemented

    def _fast_param_map(self, pred_dict, target_dict):
        """

        Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element
        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not {}, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        if len(self.param_map) == 2 and len(pred_dict) == 1 and len(target_dict) == 1:
            fast_param['pred'] = list(pred_dict.values())[0]
            fast_param['target'] = list(pred_dict.values())[0]
            return fast_param
        return fast_param

    def __call__(self, pred_dict, target_dict):
        """

        This method will call self.evaluate method.
        Before calling self.evaluate, it will first check the validity of output_dict, target_dict
            (1) whether self.evaluate has varargs, which is not supported.
            (2) whether params needed by self.evaluate is not included in output_dict,target_dict.
            (3) whether params needed by self.evaluate duplicate in pred_dict, target_dict
            (4) whether params in output_dict, target_dict are not used by evaluate.(Might cause warning)
        Besides, before passing params into self.evaluate, this function will filter out params from output_dict and
            target_dict which are not used in self.evaluate. (but if **kwargs presented in self.evaluate, no filtering
            will be conducted.)
        This function also support _fast_param_map.
        :param pred_dict: usually the output of forward or prediction function
        :param target_dict: usually features set as target..
        :return:
        """
        if not callable(self.evaluate):
            raise TypeError(f"{self.__class__.__name__}.evaluate has to be callable, not {type(self.evaluate)}.")

        fast_param = self._fast_param_map(pred_dict=pred_dict, target_dict=target_dict)
        if fast_param:
            self.evaluate(**fast_param)
            return

        if not self._checked:
            # 1. check consistence between signature and param_map
            func_spect = inspect.getfullargspec(self.evaluate)
            func_args = set([arg for arg in func_spect.args if arg != 'self'])
            for func_arg, input_arg in self.param_map.items():
                if func_arg not in func_args:
                    raise NameError(f"`{func_arg}` not in {get_func_signature(self.evaluate)}.")

            # 2. only part of the param_map are passed, left are not
            for arg in func_args:
                if arg not in self.param_map:
                    self.param_map[arg] = arg  # This param does not need mapping.
            self._evaluate_args = func_args
            self._reverse_param_map = {input_arg: func_arg for func_arg, input_arg in self.param_map.items()}

        # need to wrap inputs in dict.
        mapped_pred_dict = {}
        mapped_target_dict = {}
        duplicated = []
        for input_arg in set(list(pred_dict.keys()) + list(target_dict.keys())):
            not_duplicate_flag = 0
            if input_arg in self._reverse_param_map:
                mapped_arg = self._reverse_param_map[input_arg]
                not_duplicate_flag += 1
            else:
                mapped_arg = input_arg
            if input_arg in pred_dict:
                mapped_pred_dict[mapped_arg] = pred_dict[input_arg]
                not_duplicate_flag += 1
            if input_arg in target_dict:
                mapped_target_dict[mapped_arg] = target_dict[input_arg]
                not_duplicate_flag += 1
            if not_duplicate_flag == 3:
                duplicated.append(input_arg)

        # missing
        if not self._checked:
            check_res = _check_arg_dict_list(self.evaluate, [mapped_pred_dict, mapped_target_dict])
            # only check missing.
            # replace missing.
            missing = check_res.missing
            replaced_missing = list(missing)
            for idx, func_arg in enumerate(missing):
                # Don't delete `` in this information, nor add ``
                replaced_missing[idx] = f"{self.param_map[func_arg]}" + f"(assign to `{func_arg}` " \
                                                                        f"in `{self.__class__.__name__}`)"

            check_res = CheckRes(missing=replaced_missing,
                                 unused=check_res.unused,
                                 duplicated=duplicated,
                                 required=check_res.required,
                                 all_needed=check_res.all_needed,
                                 varargs=check_res.varargs)

            if check_res.missing or check_res.duplicated or check_res.varargs:
                raise CheckError(check_res=check_res,
                                 func_signature=get_func_signature(self.evaluate))
        refined_args = _build_args(self.evaluate, **mapped_pred_dict, **mapped_target_dict)

        self.evaluate(**refined_args)
        self._checked = True

        return


class AccuracyMetric(MetricBase):
    def __init__(self, pred=None, target=None, seq_lens=None):
        super().__init__()

        self._init_param_map(pred=pred, target=target, seq_lens=seq_lens)

        self.total = 0
        self.acc_count = 0

    def _fast_param_map(self, pred_dict, target_dict):
        """

        Only used as inner function. When the pred_dict, target is unequivocal. Don't need users to pass key_map.
            such as pred_dict has one element, target_dict has one element
        :param pred_dict:
        :param target_dict:
        :return: dict, if dict is not None, pass it to self.evaluate. Otherwise do mapping.
        """
        fast_param = {}
        targets = list(target_dict.values())
        if len(targets) == 1 and isinstance(targets[0], torch.Tensor):
            if len(pred_dict) == 1:
                pred = list(pred_dict.values())[0]
                fast_param['pred'] = pred
            elif len(pred_dict) == 2:
                pred1 = list(pred_dict.values())[0]
                pred2 = list(pred_dict.values())[1]
                if not (isinstance(pred1, torch.Tensor) and isinstance(pred2, torch.Tensor)):
                    return fast_param
                if len(pred1.size()) < len(pred2.size()) and len(pred1.size()) == 1:
                    seq_lens = pred1
                    pred = pred2
                elif len(pred1.size()) > len(pred2.size()) and len(pred2.size()) == 1:
                    seq_lens = pred2
                    pred = pred1
                else:
                    return fast_param
                fast_param['pred'] = pred
                fast_param['seq_lens'] = seq_lens
            else:
                return fast_param
            fast_param['target'] = targets[0]
        # TODO need to make sure they all have same batch_size
        return fast_param

    def evaluate(self, pred, target, seq_lens=None):
        """

        :param pred: List of (torch.Tensor, or numpy.ndarray). Element's shape can be:
                torch.Size([B,]), torch.Size([B, n_classes]), torch.Size([B, max_len]), torch.Size([B, max_len, n_classes])
        :param target: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                torch.Size([B,]), torch.Size([B,]), torch.Size([B, max_len]), torch.Size([B, max_len])
        :param seq_lens: List of (torch.Tensor, or numpy.ndarray). Element's can be:
                None, None, torch.Size([B], torch.Size([B]). ignored if masks are provided.
        :return: dict({'acc': float})
        """
        # TODO 这里报错需要更改，因为pred是啥用户并不知道。需要告知用户真实的value
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"`pred` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(pred)}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"`target` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(target)}.")

        if seq_lens is not None and not isinstance(seq_lens, torch.Tensor):
            raise TypeError(f"`seq_lens` in {get_func_signature(self.evaluate)} must be torch.Tensor,"
                            f"got {type(seq_lens)}.")

        if seq_lens is not None:
            masks = seq_lens_to_masks(seq_lens=seq_lens, float=True)
        else:
            masks = None

        if pred.size() == target.size():
            pass
        elif len(pred.size()) == len(target.size()) + 1:
            pred = pred.argmax(dim=-1)
        else:
            raise RuntimeError(f"In {get_func_signature(self.evaluate)}, when pred have "
                               f"size:{pred.size()}, target should have size: {pred.size()} or "
                               f"{pred.size()[:-1]}, got {target.size()}.")

        pred = pred.float()
        target = target.float()

        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).float() * masks.float()).item()
            self.total += torch.sum(masks.float()).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target).float()).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        evaluate_result = {'acc': round(self.acc_count / self.total, 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
        return evaluate_result


def _prepare_metrics(metrics):
    """

    Prepare list of Metric based on input
    :param metrics:
    :return: List[fastNLP.MetricBase]
    """
    _metrics = []
    if metrics:
        if isinstance(metrics, list):
            for metric in metrics:
                if isinstance(metric, type):
                    metric = metric()
                if isinstance(metric, MetricBase):
                    metric_name = metric.__class__.__name__
                    if not callable(metric.evaluate):
                        raise TypeError(f"{metric_name}.evaluate must be callable, got {type(metric.evaluate)}.")
                    if not callable(metric.get_metric):
                        raise TypeError(f"{metric_name}.get_metric must be callable, got {type(metric.get_metric)}.")
                    _metrics.append(metric)
                else:
                    raise TypeError(
                        f"The type of metric in metrics must be `fastNLP.MetricBase`, not `{type(metric)}`.")
        elif isinstance(metrics, MetricBase):
            _metrics = [metrics]
        else:
            raise TypeError(f"The type of metrics should be `list[fastNLP.MetricBase]` or `fastNLP.MetricBase`, "
                            f"got {type(metrics)}.")
    return _metrics


def accuracy_topk(y_true, y_prob, k=1):
    """Compute accuracy of y_true matching top-k probable
    labels in y_prob.

        :param y_true: ndarray, true label, [n_samples]
        :param y_prob: ndarray, label probabilities, [n_samples, n_classes]
        :param k: int, k in top-k
        :return :accuracy of top-k
    """

    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    y_true_tile = np.tile(np.expand_dims(y_true, axis=1), (1, k))
    y_match = np.any(y_pred_topk == y_true_tile, axis=-1)
    acc = np.sum(y_match) / y_match.shape[0]

    return acc


def pred_topk(y_prob, k=1):
    """Return top-k predicted labels and corresponding probabilities.


        :param y_prob: ndarray, size [n_samples, n_classes], probabilities on labels
        :param k: int, k of top-k
    :returns
        y_pred_topk: ndarray, size [n_samples, k], predicted top-k labels
        y_prob_topk: ndarray, size [n_samples, k], probabilities for top-k labels
    """

    y_pred_topk = np.argsort(y_prob, axis=-1)[:, -1:-k - 1:-1]
    x_axis_index = np.tile(
        np.arange(len(y_prob))[:, np.newaxis],
        (1, k))
    y_prob_topk = y_prob[x_axis_index, y_pred_topk]
    return y_pred_topk, y_prob_topk

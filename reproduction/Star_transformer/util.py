import fastNLP as FN
import argparse
import os
import random
import numpy
import torch


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--w_decay', type=float, required=True)
    parser.add_argument('--lr_decay', type=float, required=True)
    parser.add_argument('--bsz', type=int, required=True)
    parser.add_argument('--ep', type=int, required=True)
    parser.add_argument('--drop', type=float, required=True)
    parser.add_argument('--gpu', type=str, required=True)
    parser.add_argument('--log', type=str, default=None)
    return parser


def add_model_args(parser):
    parser.add_argument('--nhead', type=int, default=6)
    parser.add_argument('--hdim', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=300)
    return parser


def set_gpu(gpu_str):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str


def set_rng_seeds(seed=None):
    if seed is None:
        seed = numpy.random.randint(0, 65536)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # print('RNG_SEED {}'.format(seed))
    return seed


class TensorboardCallback(FN.Callback):
    """
        接受以下一个或多个字符串作为参数：
        - "model"
        - "loss"
        - "metric"
    """

    def __init__(self, *options):
        super(TensorboardCallback, self).__init__()
        args = {"model", "loss", "metric"}
        for opt in options:
            if opt not in args:
                raise ValueError(
                    "Unrecognized argument {}. Expect one of {}".format(opt, args))
        self.options = options
        self._summary_writer = None
        self.graph_added = False

    def on_train_begin(self):
        save_dir = self.trainer.save_path
        if save_dir is None:
            path = os.path.join(
                "./", 'tensorboard_logs_{}'.format(self.trainer.start_time))
        else:
            path = os.path.join(
                save_dir, 'tensorboard_logs_{}'.format(self.trainer.start_time))
        self._summary_writer = SummaryWriter(path)

    def on_batch_begin(self, batch_x, batch_y, indices):
        if "model" in self.options and self.graph_added is False:
            # tesorboardX 这里有大bug，暂时没法画模型图
            # from fastNLP.core.utils import _build_args
            # inputs = _build_args(self.trainer.model, **batch_x)
            # args = tuple([value for value in inputs.values()])
            # args = args[0] if len(args) == 1 else args
            # self._summary_writer.add_graph(self.trainer.model, torch.zeros(32, 2))
            self.graph_added = True

    def on_backward_begin(self, loss):
        if "loss" in self.options:
            self._summary_writer.add_scalar(
                "loss", loss.item(), global_step=self.trainer.step)

        if "model" in self.options:
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad:
                    self._summary_writer.add_scalar(
                        name + "_mean", param.mean(), global_step=self.trainer.step)
                    # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.trainer.step)
                    self._summary_writer.add_scalar(name + "_grad_mean", param.grad.mean(),
                                                    global_step=self.trainer.step)

    def on_valid_end(self, eval_result, metric_key):
        if "metric" in self.options:
            for name, metric in eval_result.items():
                for metric_key, metric_val in metric.items():
                    self._summary_writer.add_scalar("valid_{}_{}".format(name, metric_key), metric_val,
                                                    global_step=self.trainer.step)

    def on_train_end(self):
        self._summary_writer.close()
        del self._summary_writer

    def on_exception(self, exception):
        if hasattr(self, "_summary_writer"):
            self._summary_writer.close()
            del self._summary_writer

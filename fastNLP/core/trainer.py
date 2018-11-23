import time
from datetime import timedelta, datetime

import torch
from tensorboardX import SummaryWriter

from fastNLP.core.batch import Batch
from fastNLP.core.loss import Loss
from fastNLP.core.metrics import Evaluator
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.sampler import RandomSampler
from fastNLP.core.tester import Tester


class Trainer(object):
    """Main Training Loop

    """

    def __init__(self, train_data, model, n_epochs, batch_size, n_print,
                 dev_data=None, use_cuda=False, loss=Loss(None), save_path="./save",
                 optimizer=Optimizer("Adam", lr=0.001, weight_decay=0),
                 evaluator=Evaluator(),
                 **kwargs):
        super(Trainer, self).__init__()

        self.train_data = train_data
        self.dev_data = dev_data  # If None, No validation.
        self.model = model
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.use_cuda = bool(use_cuda)
        self.save_path = str(save_path)
        self.n_print = int(n_print)

        self.loss_func = self.model.loss if hasattr(self.model, "loss") else loss.get()
        self.optimizer = optimizer.construct_from_pytorch(self.model.parameters())
        self.evaluator = evaluator

        if self.dev_data is not None:
            valid_args = {"batch_size": self.batch_size, "save_path": self.save_path,
                          "use_cuda": self.use_cuda, "evaluator": self.evaluator}
            self.tester = Tester(**valid_args)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._summary_writer = SummaryWriter(self.save_path + 'tensorboard_logs')
        self._graph_summaried = False
        self.step = 0
        self.start_time = None  # start timestamp

        print(self.__dict__)

    def train(self):
        """Start Training.

        :return:
        """
        if torch.cuda.is_available() and self.use_cuda:
            self.model = self.model.cuda()

        self.mode(self.model, is_test=False)

        start = time.time()
        self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print("training epochs started " + self.start_time)

        epoch = 1
        while epoch <= self.n_epochs:

            data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=RandomSampler(),
                                  use_cuda=self.use_cuda)

            self._train_epoch(data_iterator, self.model, epoch, self.dev_data, start, self.n_print)

            if self.dev_data:
                self.do_validation()
            self.save_model(self.model, 'training_model_' + self.start_time)
            epoch += 1

    def _train_epoch(self, data_iterator, model, epoch, dev_data, start, n_print, **kwargs):
        """Training process in one epoch.

            kwargs should contain:
                - n_print: int, print training information every n steps.
                - start: time.time(), the starting time of this step.
                - epoch: int,
        """
        for batch_x, batch_y in data_iterator:
            prediction = self.data_forward(model, batch_x)

            # TODO: refactor self.get_loss
            loss = prediction["loss"] if "loss" in prediction else self.get_loss(prediction, batch_y)
            # acc = self._evaluator([{"predict": prediction["predict"]}], [{"truth": batch_x["truth"]}])

            self.grad_backward(loss)
            self.update()
            self._summary_writer.add_scalar("loss", loss.item(), global_step=self.step)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.step)
                    self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.step)
                    self._summary_writer.add_scalar(name + "_grad_sum", param.sum(), global_step=self.step)
            if n_print > 0 and self.step % n_print == 0:
                end = time.time()
                diff = timedelta(seconds=round(end - kwargs["start"]))
                print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                    epoch, self.step, loss.data, diff)
                print(print_output)

            self.step += 1

    def do_validation(self):
        res = self.tester.test(self.model, self.dev_data)
        for name, num in res.items():
            self._summary_writer.add_scalar("valid_{}".format(name), num, global_step=self.step)
        self.save_model(self.model, 'best_model_' + self.start_time)

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def update(self):
        """Perform weight update on a model.

        """
        self.optimizer.step()

    def data_forward(self, network, x):
        y = network(**x)
        if not self._graph_summaried:
            # self._summary_writer.add_graph(network, x, verbose=False)
            self._graph_summaried = True
        return y

    def grad_backward(self, loss):
        """Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        self.model.zero_grad()
        loss.backward()

    def get_loss(self, predict, truth):
        """Compute loss given prediction and ground truth.

        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        if isinstance(predict, dict) and isinstance(truth, dict):
            return self.loss_func(**predict, **truth)
        if len(truth) > 1:
            raise NotImplementedError("Not ready to handle multi-labels.")
        truth = list(truth.values())[0] if len(truth) > 0 else None
        return self.loss_func(predict, truth)

    def save_model(self, model, model_name, only_param=False):
        if only_param:
            torch.save(model.state_dict(), model_name)
        else:
            torch.save(model, model_name)


def best_eval_result(self, metrics):
    """Check if the current epoch yields better validation results.

    :return: bool, True means current results on dev set is the best.
    """
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        if len(metrics) == 1:
            accuracy = list(metrics.values())[0]
        else:
            accuracy = metrics[self.eval_sort_key]
    else:
        accuracy = metrics

    if accuracy > self._best_accuracy:
        self._best_accuracy = accuracy
        return True
    else:
        return False

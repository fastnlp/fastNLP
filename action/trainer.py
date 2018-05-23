from collections import namedtuple

from .action import Action
from .tester import Tester


class Trainer(Action):
    """
        Trainer for common training logic of all models
    """
    TrainConfig = namedtuple("config", ["epochs", "validate", "save_when_better", "log_per_step", "log_validation"])

    def __init__(self, train_args):
        """
        :param train_args: namedtuple
        """
        super(Trainer, self).__init__()
        self.n_epochs = train_args.epochs
        self.validate = train_args.validate
        self.save_when_better = train_args.save_when_better
        self.log_per_step = train_args.log_per_step
        self.log_validation = train_args.log_validation

    def train(self, network, train_data, dev_data):
        """
        :param network: the model controller
        :param train_data: raw data for training
        :param dev_data: raw data for validation
        :return:
        """
        train_x, train_y = network.prepare_input(train_data.train_set, train_data.train_label)

        network.mode(test=False)  # turn on the train mode

        iterations, train_batch_generator = self.batchify(train_x, train_y)

        test_args = Tester.TestConfig(save_output=True, validate_in_training=True,
                                      save_dev_input=True, save_loss=True, batch_size=16)
        evaluator = Tester(test_args)

        best_loss = 1e10
        loss_history = list()

        for epoch in range(self.n_epochs):

            network.define_optimizer()
            for step in range(iterations):
                batch_x, batch_y = train_batch_generator.__next__()

                prediction = network.data_forward(batch_x)

                loss = network.loss(batch_y, prediction)
                network.grad_backward()

                if step % self.log_per_step == 0:
                    loss_history.append(loss)
                    self.log(self.make_log(epoch, step, loss))

            #################### evaluate over dev set  ###################
            if self.validate:
                # give all controls to tester
                evaluator.test(network, dev_data)

                if self.log_validation:
                    self.log(self.make_valid_log(epoch, evaluator.loss))
                if evaluator.loss < best_loss:
                    best_loss = evaluator.loss
                    if self.save_when_better:
                        self.save_model(network)

        # finish training

    def make_log(self, *args):
        return "make a log"

    def make_valid_log(self, *args):
        return "make a valid log"

    def save_model(self, model):
        model.save()

    def load_data(self, data_name):
        print("load data")

    def load_config(self, args):
        raise NotImplementedError

    def load_dataset(self, args):
        raise NotImplementedError

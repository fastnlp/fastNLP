from collections import namedtuple

import numpy as np

from fastNLP.action import Action


class Tester(Action):
    """docstring for Tester"""

    TestConfig = namedtuple("config", ["validate_in_training", "save_dev_input", "save_output",
                                       "save_loss", "batch_size"])

    def __init__(self, test_args):
        """
        :param test_args: named tuple
        """
        super(Tester, self).__init__()
        self.validate_in_training = test_args.validate_in_training
        self.save_dev_input = test_args.save_dev_input
        self.valid_x = None
        self.valid_y = None
        self.save_output = test_args.save_output
        self.output = None
        self.save_loss = test_args.save_loss
        self.mean_loss = None
        self.batch_size = test_args.batch_size

    def test(self, network, data):
        print("testing")
        network.mode(test=True)  # turn on the testing mode
        if self.save_dev_input:
            if self.valid_x is None:
                valid_x, valid_y = network.prepare_input(data)
                self.valid_x = valid_x
                self.valid_y = valid_y
            else:
                valid_x = self.valid_x
                valid_y = self.valid_y
        else:
            valid_x, valid_y = network.prepare_input(data)

        # split into batches by self.batch_size
        iterations, test_batch_generator = self.batchify(self.batch_size, valid_x, valid_y)

        batch_output = list()
        loss_history = list()
        # turn on the testing mode of the network
        network.mode(test=True)

        for step in range(iterations):
            batch_x, batch_y = test_batch_generator.__next__()

            # forward pass from test input to predicted output
            prediction = network.data_forward(batch_x)

            loss = network.get_loss(prediction, batch_y)

            if self.save_output:
                batch_output.append(prediction.data)
            if self.save_loss:
                loss_history.append(loss)
                self.log(self.make_log(step, loss))

        if self.save_loss:
            self.mean_loss = np.mean(np.array(loss_history))
        if self.save_output:
            self.output = self.make_output(batch_output)

    @property
    def loss(self):
        return self.mean_loss

    @property
    def result(self):
        return self.output

    @staticmethod
    def make_output(batch_outputs):
        # construct full prediction with batch outputs
        return np.concatenate(batch_outputs, axis=0)

    def load_config(self, args):
        raise NotImplementedError

    def load_dataset(self, args):
        raise NotImplementedError

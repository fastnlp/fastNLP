import numpy as np

from action.action import Action


class Tester(Action):
    """docstring for Tester"""

    def __init__(self, test_args):
        """
        :param test_args: named tuple
        """
        super(Tester, self).__init__()
        self.test_args = test_args
        self.args_dict = {name: value for name, value in self.test_args.__dict__.iteritems()}
        self.mean_loss = None
        self.output = None

    def test(self, network, data):
        # transform into network input and label
        X, Y = network.prepare_input(data)

        # split into batches by self.batch_size
        iterations, test_batch_generator = self.batchify(X, Y)

        batch_output = list()
        loss_history = list()
        # turn on the testing mode of the network
        network.mode(test=True)

        for step in range(iterations):
            batch_x, batch_y = test_batch_generator.__next__()

            # forward pass from tests input to predicted output
            prediction = network.data_forward(batch_x)
            batch_output.append(prediction)

            # get the loss
            loss = network.loss(batch_y, prediction)

            loss_history.append(loss)
            self.log(self.make_log(step, loss))

        self.mean_loss = np.mean(np.array(loss_history))
        self.output = self.make_output(batch_output)

    @property
    def loss(self):
        return self.mean_loss

    @property
    def result(self):
        return self.output

    def make_output(self, batch_output):
        # construct full prediction with batch outputs
        raise NotImplementedError

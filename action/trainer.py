from .action import Action
from .tester import Tester


class Trainer(Action):
    """
        Trainer for common training logic of all models
    """

    def __init__(self, train_args):
        """
        :param train_args: namedtuple
        """
        super(Trainer, self).__init__()
        self.train_args = train_args
        # self.args_dict = {name: value for name, value in self.train_args.__dict__.iteritems()}
        self.n_epochs = self.train_args.epochs
        self.validate = self.train_args.validate
        self.save_when_better = self.train_args.save_when_better

    def train(self, network, data, dev_data):
        train_x, train_y = network.prepare_input(data.train_set, data.train_label)
        valid_x, valid_y = network.prepare_input(dev_data.valid_set, dev_data.valid_label)

        iterations, train_batch_generator = self.batchify(train_x, train_y)
        loss_history = list()
        network.mode(test=False)

        test_args = "..."
        evaluator = Tester(test_args)
        best_loss = 1e10

        for epoch in range(self.n_epochs):

            for step in range(iterations):
                batch_x, batch_y = train_batch_generator.__next__()

                prediction = network.data_forward(batch_x)

                loss = network.loss(batch_y, prediction)
                network.grad_backward()
                loss_history.append(loss)
                self.log(self.make_log(epoch, step, loss))

            #################### evaluate over dev set  ###################
            if self.validate:
                evaluator.test(network, [valid_x, valid_y])

                self.log(self.make_valid_log(epoch, evaluator.loss))
                if evaluator.loss < best_loss:
                    best_loss = evaluator.loss
                    if self.save_when_better:
                        self.save_model(network)

        # finish training

    @staticmethod
    def prepare_training(network, data):
        return network.prepare_training(data)

    def make_log(self, *args):
        print("logged")

    def make_valid_log(self, *args):
        print("logged")

    def save_model(self, model):
        print("model saved")

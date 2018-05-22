from action.action import Action
from action.tester import Tester


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
        self.args_dict = {name: value for name, value in self.train_args.__dict__.iteritems()}
        self.n_epochs = self.train_args.epochs
        self.validate = True
        self.save_when_better = True

    def train(self, network, data, dev_data):
        X, Y = network.prepare_input(data)

        iterations, train_batch_generator = self.batchify(X, Y)
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

            # evaluate over dev set
            if self.validate:
                evaluator.test(network, dev_data)
                self.log(self.make_valid_log(epoch, evaluator.loss))
                if evaluator.loss < best_loss:
                    best_loss = evaluator.loss
                    if self.save_when_better:
                        self.save_model(network)

        # finish training

    def make_log(self, *args):
        raise NotImplementedError

    def make_valid_log(self, *args):
        raise NotImplementedError

    def save_model(self, model):
        raise NotImplementedError

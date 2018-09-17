import numpy as np
import torch

from fastNLP.core.action import RandomSampler
from fastNLP.core.batch import Batch
from fastNLP.saver.logger import create_logger

logger = create_logger(__name__, "./train_test.log")


class BaseTester(object):
    """An collection of model inference and evaluation of performance, used over validation/dev set and test set. """

    def __init__(self, **kwargs):
        """
        :param kwargs: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(BaseTester, self).__init__()
        """
            "default_args" provides default value for important settings. 
            The initialization arguments "kwargs" with the same key (name) will override the default value. 
            "kwargs" must have the same type as "default_args" on corresponding keys. 
            Otherwise, error will raise.
        """
        default_args = {"save_output": False,  # collect outputs of validation set
                        "save_loss": False,  # collect losses in validation
                        "save_best_dev": False,  # save best model during validation
                        "batch_size": 8,
                        "use_cuda": True,
                        "pickle_path": "./save/",
                        "model_name": "dev_best_model.pkl",
                        "print_every_step": 1,
                        }
        """
            "required_args" is the collection of arguments that users must pass to Trainer explicitly. 
            This is used to warn users of essential settings in the training. 
            Specially, "required_args" does not have default value, so they have nothing to do with "default_args".
        """
        required_args = {"task"  # one of ("seq_label", "text_classify")
                         }

        for req_key in required_args:
            if req_key not in kwargs:
                logger.error("Tester lacks argument {}".format(req_key))
                raise ValueError("Tester lacks argument {}".format(req_key))
        self._task = kwargs["task"]

        for key in default_args:
            if key in kwargs:
                if isinstance(kwargs[key], type(default_args[key])):
                    default_args[key] = kwargs[key]
                else:
                    msg = "Argument %s type mismatch: expected %s while get %s" % (
                        key, type(default_args[key]), type(kwargs[key]))
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                # BaseTester doesn't care about extra arguments
                pass
        print(default_args)

        self.save_output = default_args["save_output"]
        self.save_best_dev = default_args["save_best_dev"]
        self.save_loss = default_args["save_loss"]
        self.batch_size = default_args["batch_size"]
        self.pickle_path = default_args["pickle_path"]
        self.use_cuda = default_args["use_cuda"]
        self.print_every_step = default_args["print_every_step"]

        self._model = None
        self.eval_history = []  # evaluation results of all batches
        self.batch_output = []  # outputs of all batches

    def test(self, network, dev_data):
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
        else:
            self._model = network

        # turn on the testing mode; clean up the history
        self.mode(network, is_test=True)
        self.eval_history.clear()
        self.batch_output.clear()

        data_iterator = Batch(dev_data, self.batch_size, sampler=RandomSampler(), use_cuda=self.use_cuda)
        step = 0

        for batch_x, batch_y in data_iterator:
            with torch.no_grad():
                prediction = self.data_forward(network, batch_x)
                eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                self.batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

            print_output = "[test step {}] {}".format(step, eval_results)
            logger.info(print_output)
            if self.print_every_step > 0 and step % self.print_every_step == 0:
                print(self.make_eval_output(prediction, eval_results))
            step += 1

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def data_forward(self, network, x):
        """A forward pass of the model. """
        y = network(**x)
        return y

    def evaluate(self, predict, truth):
        """Compute evaluation metrics.

        :param predict: Tensor
        :param truth: Tensor
        :return eval_results: can be anything. It will be stored in self.eval_history
        """
        if "label_seq" in truth:
            truth = truth["label_seq"]
        elif "label" in truth:
            truth = truth["label"]
        else:
            raise NotImplementedError("Unknown key {} in batch_y.".format(truth.keys()))

        if self._task == "seq_label":
            return self._seq_label_evaluate(predict, truth)
        elif self._task == "text_classify":
            return self._text_classify_evaluate(predict, truth)
        else:
            raise NotImplementedError("Unknown task type {}.".format(self._task))

    def _seq_label_evaluate(self, predict, truth):
        batch_size, max_len = predict.size(0), predict.size(1)
        loss = self._model.loss(predict, truth) / batch_size
        prediction = self._model.prediction(predict)
        # pad prediction to equal length
        for pred in prediction:
            if len(pred) < max_len:
                pred += [0] * (max_len - len(pred))
        results = torch.Tensor(prediction).view(-1, )

        # make sure "results" is in the same device as "truth"
        results = results.to(truth)
        accuracy = torch.sum(results == truth.view((-1,))).to(torch.float) / results.shape[0]
        return [float(loss), float(accuracy)]

    def _text_classify_evaluate(self, y_logit, y_true):
        y_prob = torch.nn.functional.softmax(y_logit, dim=-1)
        return [y_prob, y_true]

    @property
    def metrics(self):
        """Compute and return metrics.
        Use self.eval_history to compute metrics over the whole dev set.
        Please refer to metrics.py for common metric functions.

        :return : variable number of outputs
        """
        if self._task == "seq_label":
            return self._seq_label_metrics
        elif self._task == "text_classify":
            return self._text_classify_metrics
        else:
            raise NotImplementedError("Unknown task type {}.".format(self._task))

    @property
    def _seq_label_metrics(self):
        batch_loss = np.mean([x[0] for x in self.eval_history])
        batch_accuracy = np.mean([x[1] for x in self.eval_history])
        return batch_loss, batch_accuracy

    @property
    def _text_classify_metrics(self):
        y_prob, y_true = zip(*self.eval_history)
        y_prob = torch.cat(y_prob, dim=0)
        y_pred = torch.argmax(y_prob, dim=-1)
        y_true = torch.cat(y_true, dim=0)
        acc = float(torch.sum(y_pred == y_true)) / len(y_true)
        return y_true.cpu().numpy(), y_prob.cpu().numpy(), acc

    def show_metrics(self):
        """Customize evaluation outputs in Trainer.
        Called by Trainer to print evaluation results on dev set during training.
        Use self.metrics to fetch available metrics.

        :return print_str: str
        """
        loss, accuracy = self.metrics
        return "dev loss={:.2f}, accuracy={:.2f}".format(loss, accuracy)

    def make_eval_output(self, predictions, eval_results):
        """Customize Tester outputs.

        :param predictions: Tensor
        :param eval_results: Tensor
        :return: str, to be printed.
        """
        return self.show_metrics()


class SeqLabelTester(BaseTester):
    def __init__(self, **test_args):
        test_args.update({"task": "seq_label"})
        print(
            "[FastNLP Warning] SeqLabelTester will be deprecated. Please use Tester with argument 'task'='seq_label'.")
        super(SeqLabelTester, self).__init__(**test_args)


class ClassificationTester(BaseTester):
    def __init__(self, **test_args):
        test_args.update({"task": "seq_label"})
        print(
            "[FastNLP Warning] ClassificationTester will be deprecated. Please use Tester with argument 'task'='text_classify'.")
        super(ClassificationTester, self).__init__(**test_args)

import numpy as np
import torch

from fastNLP.core.action import Action
from fastNLP.core.action import RandomSampler, Batchifier
from fastNLP.modules import utils
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
            Obviously, "required_args" is the subset of "default_args". 
            The value in "default_args" to the keys in "required_args" is simply for type check. 
        """
        # add required arguments here
        required_args = {}

        for req_key in required_args:
            if req_key not in kwargs:
                logger.error("Tester lacks argument {}".format(req_key))
                raise ValueError("Tester lacks argument {}".format(req_key))

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
        self.eval_score = 0  # evaluation score of all batches
        self.result = []
        self.lable = []
        self.all_mask = []

    def test(self, network, dev_data):
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
        else:
            self._model = network

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)
        self.eval_score = 0
        self.result.clear()
        self.lable.clear()
        self.all_mask.clear()

        iterator = iter(Batchifier(RandomSampler(dev_data), self.batch_size, drop_last=False))


        for batch_x, batch_y in self.make_batch(iterator):
            with torch.no_grad():
                prediction = self.data_forward(network, batch_x)
                eval_result = self.evaluate(prediction,batch_y)
                self.result.extend(eval_result[0])
                self.lable.extend(eval_result[1])
                if len(eval_result)==3:
                    self.all_mask.extend(eval_result[2])#if have mask
        self.make_eval_output()

    def mode(self, model, test):
        """Train mode or Test mode. This is for PyTorch currently.
        :param model: a PyTorch model
        :param test: bool, whether in test mode.
        """
        Action.mode(model, test)

    def data_forward(self, network, x):
        """A forward pass of the model. """
        raise NotImplementedError

    def evaluate(self, predict, truth):
        """Compute evaluation metrics.
        :param predict: Tensor
        :param truth: Tensor
        :return eval_results:list of numpy,  predicted tags and true tags
        """
        raise NotImplementedError

    def metrics(self):
        """Compute and return metrics.
        Use self.eval_history to compute metrics over the whole dev set.
        Please refer to metrics.py for common metric functions.
        :return : variable number of outputs
        """

        best_result = self.eval_score
        return best_result

    def show_metrics(self):
        """Customize evaluation outputs in Trainer.
        Called by Trainer to print evaluation results on dev set during training.
        Use self.metrics to fetch available metrics.
        :return print_str: str
        """
        loss, accuracy = self.metrics()
        return "dev loss={:.2f}, accuracy={:.2f}".format(loss, accuracy)

    def make_batch(self, iterator):
        raise NotImplementedError

    def make_eval_output(self,):
        """
        Customize Tester outputs. you can overload to change the Evaluation standard
        """
        predictions=self.result
        y_label=self.lable
        if self.all_mask:
            mask=self.all_mask
        else:
            mask=[None]*len(predictions)
        Acc_num, All_num, Pred_num, Label_num, Pred_right_num = 0,0,0,0,0#Acc_num is the number of exact match right preds
                                                                         #All_num is the sum number of samples
                                                                         #Pred_num is the number of predicted nonzero labels
                                                                         #Label_num is the number of nonzero labels
                                                                         #Pred_right_num is the number of predicted right nonzero labels
        for p,y,m in zip(predictions,y_label,mask):
            acc_num,all_num, pred_num, label_num, pred_right_num=Action.get_real(p,y,m)#if only care about acc, ignore pred_num, label_num, pred_right_num.
                                                                                     # if lable 0 is negtive sample, pred_num, label_num, pred_right_num work too.
            Acc_num+=acc_num
            All_num+=all_num
            Pred_num+=pred_num
            Label_num+=label_num
            Pred_right_num+=pred_right_num

        accuracy=Acc_num/All_num
        precision=Pred_right_num/Pred_num
        recall=Pred_right_num/Label_num
        f1_score=2*precision*recall/(precision+recall)
        self.print_output(accuracy,precision,recall,f1_score)

    def print_output(self,accuracy,precision,recall,f1_score):
        """
        choose which score to print and evaluate
        """
        self.eval_score = accuracy
        print_output = "accuracy:{:.3}, precision:{:.3}, recall:{:.3}, f1_score:{:.3}".format(accuracy,precision,recall,f1_score)
        logger.info(print_output)
        print(print_output)

class SeqLabelTester(BaseTester):
    """Tester for sequence labeling.

    """

    def __init__(self, **test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(SeqLabelTester, self).__init__(**test_args)
        self.max_len = None
        self.mask = None
        self.seq_len = None

    def data_forward(self, network, inputs):
        """This is only for sequence labeling with CRF decoder.
        :param network: a PyTorch model
        :param inputs: tuple of (x, seq_len)
                        x: Tensor of shape [batch_size, max_len], where max_len is the maximum length of the mini-batch
                            after padding.
                        seq_len: list of int, the lengths of sequences before padding.
        :return y: Tensor of shape [batch_size, max_len]
        """
        if not isinstance(inputs, tuple):
            raise RuntimeError("output_length must be true for sequence modeling.")
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]
        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)
        if torch.cuda.is_available() and self.use_cuda:
            mask = mask.cuda()
        self.mask = mask
        self.seq_len = seq_len
        y = network(x)
        return y

    def evaluate(self, predict, truth):
        """Compute tags.
        :param predict: Tensor, [batch_size, max_len, tag_size]
        :param truth: Tensor, [batch_size, max_len]
        :return:
        """

        prediction = self._model.prediction(predict, self.mask)
        prediction = prediction.cpu().numpy()
        truth = truth.cpu().numpy()
        return [Action.cut_pad(prediction,self.seq_len),Action.cut_pad(truth,self.seq_len)]

    def make_batch(self, iterator):
        return Action.make_batch(iterator, use_cuda=self.use_cuda, output_length=True)

      
class AdvSeqLabelTester(BaseTester):
    """
        Tester for sequence labeling.(multi-feature)
    """
    def __init__(self, **test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"

        """
        super(AdvSeqLabelTester, self).__init__(**test_args)
        self.max_len = None
        self.mask = None
        self.seq_len = None

    def data_forward(self, network, inputs):
        if not isinstance(inputs, tuple):
            raise RuntimeError("output_length must be true for sequence modeling. Receive {}".format(type(inputs[0])))
        # unpack the returned value from make_batch
        x, seq_len, all_fea = inputs[0], inputs[1], inputs[2]

        value_ph=all_fea[1]
        entity=all_fea[0]
        self.seq_len=seq_len

        self.mask = entity.ge(1)#my task need this mask

        y = network(x,entity,value_ph)
        return y

    def evaluate(self, predict, truth):
        """Compute tags.
        :param predict: Tensor, [batch_size, max_len, tag_size]
        :param truth: Tensor, [batch_size, max_len]
        :return:
        """

        prediction = self._model.prediction(predict, self.mask)
        prediction = prediction.cpu().numpy()
        truth = truth.cpu().numpy()
        return [Action.cut_pad(prediction,self.seq_len),Action.cut_pad(truth,self.seq_len),Action.cut_pad(self.mask,self.seq_len)]

    def make_batch(self, iterator):
        return Action.adv_make_batch(iterator, output_length=True, use_cuda=self.use_cuda)


class ClassificationTester(BaseTester):
    """Tester for classification."""

    def __init__(self, **test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method.
            can be accessed by "test_args["key_str"]"
        """
        super(ClassificationTester, self).__init__(**test_args)

    def make_batch(self, iterator, max_len=None):
        return Action.make_batch(iterator, use_cuda=self.use_cuda, max_len=max_len)

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def evaluate(self, y_logit, y_true):
        """Return y_pred and y_true."""
        y_prob = torch.argmax(y_logit,-1)
        return [y_prob, y_true]

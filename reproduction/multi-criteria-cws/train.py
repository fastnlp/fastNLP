from fastNLP import (Trainer, Tester, Callback, GradientClipCallback, LRScheduler, SpanFPreRecMetric)
import torch
import torch.cuda
from torch.optim import Adam, SGD
from argparse import ArgumentParser
import logging
from .utils import set_seed


class LoggingCallback(Callback):
    def __init__(self, filepath=None):
        super().__init__()
        # create file handler and set level to debug
        if filepath is not None:
            file_handler = logging.FileHandler(filepath, "a")
        else:
            file_handler = logging.StreamHandler()

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                              datefmt='%m/%d/%Y %H:%M:%S'))

        # create logger and set level to debug
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(file_handler)
        self.log_writer = logger

    def on_backward_begin(self, loss):
        if self.step % self.trainer.print_every == 0:
            self.log_writer.info(
                'Step/Epoch {}/{}: Loss {}'.format(self.step, self.epoch, loss.item()))

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        self.log_writer.info(
            'Step/Epoch {}/{}: Eval result {}'.format(self.step, self.epoch, eval_result))

    def on_backward_end(self):
        pass


def main():
    parser = ArgumentParser()
    register_args(parser)
    args = parser.parse_known_args()[0]

    set_seed(args.seed)
    if args.train:
        train(args)
    if args.eval:
        evaluate(args)

def get_optim(args):
    name = args.optim.strip().split(' ')[0].lower()
    p = args.optim.strip()
    l = p.find('(')
    r = p.find(')')
    optim_args = eval('dict({})'.format(p[[l+1,r]]))
    if name == 'sgd':
        return SGD(**optim_args)
    elif name == 'adam':
        return Adam(**optim_args)
    else:
        raise ValueError(args.optim)

def load_model_from_path(args):
    pass

def train(args):
    data = get_data(args)
    train_data = data['train']
    dev_data = data['dev']
    model = get_model(args)
    optimizer = get_optim(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    callbacks = []
    trainer = Trainer(
        train_data=train_data,
        model=model,
        optimizer=optimizer,
        loss=None,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        num_workers=4,
        metrics=SpanFPreRecMetric(
            tag_vocab=data['tag_vocab'], encoding_type=data['encoding_type'],
            ignore_labels=data['ignore_labels']),
        metric_key='f1',
        dev_data=dev_data,
        save_path=args.save_path,
        device=device,
        callbacks=callbacks,
        check_code_level=-1,
    )

    print(trainer.train())



def evaluate(args):
    data = get_data(args)
    test_data = data['test']
    model = load_model_from_path(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tester = Tester(
        data=test_data, model=model, batch_size=args.batch_size,
        num_workers=2, device=device,
        metrics=SpanFPreRecMetric(
            tag_vocab=data['tag_vocab'], encoding_type=data['encoding_type'],
            ignore_labels=data['ignore_labels']),
    )
    print(tester.test())

def register_args(parser):
    parser.add_argument('--optim', type=str, default='adam (lr=2e-3, weight_decay=0.0)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42, help='rng seed')

def get_model(args):
    pass

def get_data(args):
    return torch.load(args.data_path)

if __name__ == '__main__':
    main()

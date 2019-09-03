import sys
sys.path.append('../..')

import torch
from torch.optim import Adam

from fastNLP.core.callback import Callback, GradientClipCallback
from fastNLP.core.trainer import Trainer

from fastNLP.io.pipe.coreference import CoreferencePipe

from reproduction.coreference_resolution.model.config import Config
from reproduction.coreference_resolution.model.model_re import Model
from reproduction.coreference_resolution.model.softmax_loss import SoftmaxLoss
from reproduction.coreference_resolution.model.metric import CRMetric
from fastNLP import SequentialSampler
from fastNLP import cache_results


# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

class LRCallback(Callback):
    def __init__(self, parameters, decay_rate=1e-3):
        super().__init__()
        self.paras = parameters
        self.decay_rate = decay_rate

    def on_step_end(self):
        if self.step % 100 == 0:
            for para in self.paras:
                para['lr'] = para['lr'] * (1 - self.decay_rate)


if __name__ == "__main__":
    config = Config()

    print(config)

    @cache_results('cache.pkl')
    def cache():
        bundle = CoreferencePipe(Config()).process_from_file({'train': config.train_path, 'dev': config.dev_path,'test': config.test_path})
        return bundle
    data_info = cache()
    print("数据集划分：\ntrain:", str(len(data_info.datasets["train"])),
          "\ndev:" + str(len(data_info.datasets["dev"])) + "\ntest:" + str(len(data_info.datasets["test"])))
    # print(data_info)
    model = Model(data_info.vocabs['vocab'], config)
    print(model)

    loss = SoftmaxLoss()

    metric = CRMetric()

    optim = Adam(model.parameters(), lr=config.lr)

    lr_decay_callback = LRCallback(optim.param_groups, config.lr_decay)

    trainer = Trainer(model=model, train_data=data_info.datasets["train"], dev_data=data_info.datasets["dev"],
                      loss=loss, metrics=metric, check_code_level=-1,sampler=None,
                      batch_size=1, device=torch.device("cuda:" + config.cuda), metric_key='f', n_epochs=config.epoch,
                      optimizer=optim,
                      save_path='/remote-home/xxliu/pycharm/fastNLP/fastNLP/reproduction/coreference_resolution/save',
                      callbacks=[lr_decay_callback, GradientClipCallback(clip_value=5)])
    print()

    trainer.train()

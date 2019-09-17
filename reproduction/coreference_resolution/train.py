
import torch
from torch.optim import Adam

from fastNLP.core.callback import Callback, GradientClipCallback
from fastNLP.core.trainer import Trainer

from fastNLP.io.pipe.coreference import CoReferencePipe
from fastNLP.core.const import Const

from reproduction.coreference_resolution.model.config import Config
from reproduction.coreference_resolution.model.model_re import Model
from reproduction.coreference_resolution.model.softmax_loss import SoftmaxLoss
from reproduction.coreference_resolution.model.metric import CRMetric


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

    def cache():
        bundle = CoReferencePipe(config).process_from_file({'train': config.train_path, 'dev': config.dev_path,
                                                            'test': config.test_path})
        return bundle
    data_bundle = cache()
    print(data_bundle)
    model = Model(data_bundle.get_vocab(Const.INPUTS(0)), config)
    print(model)

    loss = SoftmaxLoss()

    metric = CRMetric()

    optim = Adam(model.parameters(), lr=config.lr)

    lr_decay_callback = LRCallback(optim.param_groups, config.lr_decay)

    trainer = Trainer(model=model, train_data=data_bundle.datasets["train"], dev_data=data_bundle.datasets["dev"],
                      loss=loss, metrics=metric, check_code_level=-1, sampler=None,
                      batch_size=1, device=torch.device("cuda:" + config.cuda) if torch.cuda.is_available() else None,
                      metric_key='f', n_epochs=config.epoch,
                      optimizer=optim,
                      save_path=None,
                      callbacks=[lr_decay_callback, GradientClipCallback(clip_value=5)])
    print()

    trainer.train()

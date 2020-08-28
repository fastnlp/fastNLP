from fastNLP.io import OntoNotesNERPipe
from fastNLP.core.callback import LRScheduler
from fastNLP import GradientClipCallback
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from fastNLP import Const
from fastNLP import BucketSampler
from fastNLP import SpanFPreRecMetric
from fastNLP import Trainer, Tester
from fastNLP.core.metrics import MetricBase
from reproduction.sequence_labelling.ner.model.dilated_cnn import IDCNN
from fastNLP.core.utils import Option
from fastNLP.embeddings import StaticEmbedding
from fastNLP.core.utils import cache_results
import torch.cuda
import os

encoding_type = 'bioes'


def get_path(path):
    return os.path.join(os.environ['HOME'], path)


ops = Option(
    batch_size=128,
    num_epochs=100,
    lr=3e-4,
    repeats=3,
    num_layers=3,
    num_filters=400,
    use_crf=False,
    gradient_clip=5,
)

@cache_results('ontonotes-case-cache')
def load_data():
    print('loading data')
    data = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(
        paths = get_path('workdir/datasets/ontonotes-v4'))
    print('loading embedding')
    word_embed = StaticEmbedding(vocab=data.vocabs[Const.INPUT],
                                 model_dir_or_name='en-glove-840b-300',
                                 requires_grad=True)
    return data, [word_embed]

data, embeds = load_data()
print(data)
print(data.datasets['train'][0])
print(list(data.vocabs.keys()))

# for ds in data.datasets.values():
#     ds.rename_field('cap_words', 'chars')
#     ds.set_input('chars')

word_embed = embeds[0]
word_embed.embedding.weight.data /= word_embed.embedding.weight.data.std()

# char_embed = CNNCharEmbedding(data.vocabs['cap_words'])
char_embed = None
# for ds in data.datasets:
#     ds.rename_field('')

print(data.vocabs[Const.TARGET].word2idx)

model = IDCNN(init_embed=word_embed,
              char_embed=char_embed,
              num_cls=len(data.vocabs[Const.TARGET]),
              repeats=ops.repeats,
              num_layers=ops.num_layers,
              num_filters=ops.num_filters,
              kernel_size=3,
              use_crf=ops.use_crf, use_projection=True,
              block_loss=True,
              input_dropout=0.5, hidden_dropout=0.2, inner_dropout=0.2)

print(model)

callbacks = [GradientClipCallback(clip_value=ops.gradient_clip, clip_type='value'),]
metrics = []
metrics.append(
    SpanFPreRecMetric(
        tag_vocab=data.vocabs[Const.TARGET], encoding_type=encoding_type,
        pred=Const.OUTPUT, target=Const.TARGET, seq_len=Const.INPUT_LEN,
    )
)

class LossMetric(MetricBase):
    def __init__(self, loss=None):
        super(LossMetric, self).__init__()
        self._init_param_map(loss=loss)
        self.total_loss = 0.0
        self.steps = 0

    def evaluate(self, loss):
        self.total_loss += float(loss)
        self.steps += 1

    def get_metric(self, reset=True):
        result = {'loss': self.total_loss / (self.steps + 1e-12)}
        if reset:
            self.total_loss = 0.0
            self.steps = 0
        return result

metrics.append(
    LossMetric(loss=Const.LOSS)
)

optimizer = Adam(model.parameters(), lr=ops.lr, weight_decay=0)
scheduler = LRScheduler(LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch)))
callbacks.append(scheduler)
# callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 15)))
# optimizer = SWATS(model.parameters(), verbose=True)
# optimizer = Adam(model.parameters(), lr=0.005)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

trainer = Trainer(train_data=data.datasets['train'], model=model, optimizer=optimizer,
                  sampler=BucketSampler(num_buckets=50, batch_size=ops.batch_size),
                  device=device, dev_data=data.datasets['dev'], batch_size=ops.batch_size,
                  metrics=metrics,
                  check_code_level=-1,
                  callbacks=callbacks, num_workers=2, n_epochs=ops.num_epochs)
trainer.train()

torch.save(model, 'idcnn.pt')

tester = Tester(
    data=data.datasets['test'],
    model=model,
    metrics=metrics,
    batch_size=ops.batch_size,
    num_workers=2,
    device=device
)
tester.test()


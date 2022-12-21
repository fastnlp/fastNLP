"""
此模块可以非常方便的测试模型。
若你的模型属于：文本分类，序列标注，自然语言推理（NLI），可以直接使用此模块测试
若模型不属于上述类别，也可以自己准备假数据，设定loss和metric进行测试

此模块的测试仅保证模型能使用fastNLP进行训练和测试，不测试模型实际性能

Example::

    # import 全大写变量...
    from model_runner import *

    # 测试一个文本分类模型
    init_emb = (VOCAB_SIZE, 50)
    model = SomeModel(init_emb, num_cls=NUM_CLS)
    RUNNER.run_model_with_task(TEXT_CLS, model)

    # 序列标注模型
    RUNNER.run_model_with_task(POS_TAGGING, model)

    # NLI模型
    RUNNER.run_model_with_task(NLI, model)

    # 自定义模型
    RUNNER.run_model(model, data=get_mydata(),
     loss=Myloss(), metrics=Mymetric())
"""
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    from torch import optim
from fastNLP import Trainer, Evaluator, DataSet, Callback
from fastNLP import Accuracy
from random import randrange
from fastNLP import TorchDataLoader

VOCAB_SIZE = 100
NUM_CLS = 100
MAX_LEN = 10
N_SAMPLES = 100
N_EPOCHS = 1
BATCH_SIZE = 5

TEXT_CLS = 'text_cls'
POS_TAGGING = 'pos_tagging'
NLI = 'nli'

class ModelRunner():
    class Checker(Callback):
        def on_backward_begin(self, trainer, outputs):
            assert outputs['loss'].to('cpu').numpy().isfinate()

    def gen_seq(self, length, vocab_size):
        """generate fake sequence indexes with given length"""
        # reserve 0 for padding
        return [randrange(1, vocab_size) for _ in range(length)]

    def gen_var_seq(self, max_len, vocab_size):
        """generate fake sequence indexes in variant length"""
        length = randrange(3, max_len) # at least 3 words in a seq
        return self.gen_seq(length, vocab_size)

    def prepare_text_classification_data(self):
        index = 'index'
        ds = DataSet({index: list(range(N_SAMPLES))})
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name='words')
        ds.apply_field(lambda x: randrange(NUM_CLS),
                       field_name=index, new_field_name='target')
        ds.apply_field(len, 'words', 'seq_len')
        dl = TorchDataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def prepare_pos_tagging_data(self):
        index = 'index'
        ds = DataSet({index: list(range(N_SAMPLES))})
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name='words')
        ds.apply_field(lambda x: self.gen_seq(len(x), NUM_CLS),
                       field_name='words', new_field_name='target')
        ds.apply_field(len, 'words', 'seq_len')
        dl = TorchDataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def prepare_nli_data(self):
        index = 'index'
        ds = DataSet({index: list(range(N_SAMPLES))})
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name='words1')
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name='words2')
        ds.apply_field(lambda x: randrange(NUM_CLS),
                       field_name=index, new_field_name='target')
        ds.apply_field(len, 'words1', 'seq_len1')
        ds.apply_field(len, 'words2', 'seq_len2')
        dl = TorchDataLoader(ds, batch_size=BATCH_SIZE)
        return dl

    def run_text_classification(self, model, data=None):
        if data is None:
            data = self.prepare_text_classification_data()
        metric = Accuracy()
        self.run_model(model, data, metric)

    def run_pos_tagging(self, model, data=None):
        if data is None:
            data = self.prepare_pos_tagging_data()
        metric = Accuracy()
        self.run_model(model, data, metric)

    def run_nli(self, model, data=None):
        if data is None:
            data = self.prepare_nli_data()
        metric = Accuracy()
        self.run_model(model, data, metric)

    def run_model(self, model, data, metrics):
        """run a model, test if it can run with fastNLP"""
        print('testing model:', model.__class__.__name__)
        tester = Evaluator(model, data, metrics={'metric': metrics}, driver='torch')
        before_train = tester.run()
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        trainer = Trainer(model, driver='torch', train_dataloader=data,
                          n_epochs=N_EPOCHS, optimizers=optimizer)
        trainer.run()
        after_train = tester.run()
        for metric_name, v1 in before_train.items():
            assert metric_name in after_train
            # # at least we can sure model params changed, even if we don't know performance
            # v2 = after_train[metric_name]
            # assert v1 != v2

    def run_model_with_task(self, task, model):
        """run a model with certain task"""
        TASKS = {
            TEXT_CLS: self.run_text_classification,
            POS_TAGGING: self.run_pos_tagging,
            NLI: self.run_nli,
        }
        assert task in TASKS
        TASKS[task](model)

RUNNER = ModelRunner()

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
from fastNLP import Trainer, Tester, DataSet, Callback
from fastNLP import AccuracyMetric
from fastNLP import CrossEntropyLoss
from fastNLP.core.const import Const as C
from random import randrange

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
        def on_backward_begin(self, loss):
            assert loss.to('cpu').numpy().isfinate()

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
                       field_name=index, new_field_name=C.INPUT,
                       is_input=True)
        ds.apply_field(lambda x: randrange(NUM_CLS),
                       field_name=index, new_field_name=C.TARGET,
                       is_target=True)
        ds.apply_field(len, C.INPUT, C.INPUT_LEN,
                       is_input=True)
        return ds

    def prepare_pos_tagging_data(self):
        index = 'index'
        ds = DataSet({index: list(range(N_SAMPLES))})
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name=C.INPUT,
                       is_input=True)
        ds.apply_field(lambda x: self.gen_seq(len(x), NUM_CLS),
                       field_name=C.INPUT, new_field_name=C.TARGET,
                       is_target=True)
        ds.apply_field(len, C.INPUT, C.INPUT_LEN,
                       is_input=True, is_target=True)
        return ds

    def prepare_nli_data(self):
        index = 'index'
        ds = DataSet({index: list(range(N_SAMPLES))})
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name=C.INPUTS(0),
                       is_input=True)
        ds.apply_field(lambda x: self.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                       field_name=index, new_field_name=C.INPUTS(1),
                       is_input=True)
        ds.apply_field(lambda x: randrange(NUM_CLS),
                       field_name=index, new_field_name=C.TARGET,
                       is_target=True)
        ds.apply_field(len, C.INPUTS(0), C.INPUT_LENS(0),
                       is_input=True, is_target=True)
        ds.apply_field(len, C.INPUTS(1), C.INPUT_LENS(1),
                       is_input = True, is_target = True)
        ds.set_input(C.INPUTS(0), C.INPUTS(1))
        ds.set_target(C.TARGET)
        return ds

    def run_text_classification(self, model, data=None):
        if data is None:
            data = self.prepare_text_classification_data()
        loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
        metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
        self.run_model(model, data, loss, metric)

    def run_pos_tagging(self, model, data=None):
        if data is None:
            data = self.prepare_pos_tagging_data()
        loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET, padding_idx=0)
        metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET, seq_len=C.INPUT_LEN)
        self.run_model(model, data, loss, metric)

    def run_nli(self, model, data=None):
        if data is None:
            data = self.prepare_nli_data()
        loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
        metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
        self.run_model(model, data, loss, metric)

    def run_model(self, model, data, loss, metrics):
        """run a model, test if it can run with fastNLP"""
        print('testing model:', model.__class__.__name__)
        tester = Tester(data=data, model=model, metrics=metrics,
                        batch_size=BATCH_SIZE, verbose=0)
        before_train = tester.test()
        trainer = Trainer(model=model, train_data=data, dev_data=None,
                          n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                          loss=loss,
                          save_path=None,
                          use_tqdm=False)
        trainer.train(load_best_model=False)
        after_train = tester.test()
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

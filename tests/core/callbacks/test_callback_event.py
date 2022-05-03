import pytest
from functools import reduce

from fastNLP.core.callbacks.callback_event import Event, Filter



class TestFilter:
    def test_every_filter(self):
        # every = 10
        @Filter(every=10)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [w-1 for w in range(10, 101, 10)]

        # every = 1
        @Filter(every=1)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == list(range(100))

    def test_once_filter(self):
        # once = 10
        @Filter(once=10)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [9]


    def test_extract_filter_from_fn(self):
        @Filter(every=10)
        def _fn(data):
            return data

        _filter_num_called = []
        _filter_num_executed = []
        for i in range(100):
            cu_res = _fn(i)
            _filter = _fn.__fastNLP_filter__
            _filter_num_called.append(_filter.num_called)
            _filter_num_executed.append(_filter.num_executed)
        assert _filter_num_called == list(range(1, 101))
        assert _filter_num_executed == [0]*9 + reduce(lambda x, y: x+y, [[w]*10 for w in range(1, 10)]) + [10]

        def _fn(data):
            return data
        assert not hasattr(_fn, "__fastNLP_filter__")

    def test_filter_state_dict(self):
        # every = 10
        @Filter(every=10)
        def _fn(data):
            return data

        _res = []
        for i in range(50):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [w - 1 for w in range(10, 51, 10)]

        # 保存状态
        state = _fn.__fastNLP_filter__.state_dict()
        # 加载状态
        _fn.__fastNLP_filter__.load_state_dict(state)

        _res = []
        for i in range(50, 100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [w - 1 for w in range(60, 101, 10)]


@pytest.mark.torch
def test_filter_fn_torch():
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from fastNLP.core.controllers.trainer import Trainer
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
    from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)
    optimizer = SGD(model.parameters(), lr=0.0001)
    dataset = TorchNormalDataset_Classification(num_labels=3, feature_dimension=10)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    trainer = Trainer(model=model, driver="torch", device="cpu", train_dataloader=dataloader, optimizers=optimizer)
    def filter_fn(filter, trainer):
        if trainer.__heihei_test__ == 10:
            return True
        return False

    @Filter(filter_fn=filter_fn)
    def _fn(trainer, data):
        return data

    _res = []
    for i in range(100):
        trainer.__heihei_test__ = i
        cu_res = _fn(trainer, i)
        if cu_res is not None:
            _res.append(cu_res)
    assert _res == [10]


class TestCallbackEvents:
    def test_every(self):

        # 这里是什么样的事件是不影响的，因为我们是与 Trainer 拆分开了进行测试；
        event_state = Event.on_train_begin()  # 什么都不输入是应当默认 every=1；
        @Filter(every=event_state.every, once=event_state.once, filter_fn=event_state.filter_fn)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == list(range(100))

        event_state = Event.on_train_begin(every=10)
        @Filter(every=event_state.every, once=event_state.once, filter_fn=event_state.filter_fn)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [w - 1 for w in range(10, 101, 10)]

    def test_once(self):
        event_state = Event.on_train_begin(once=10)

        @Filter(once=event_state.once)
        def _fn(data):
            return data

        _res = []
        for i in range(100):
            cu_res = _fn(i)
            if cu_res is not None:
                _res.append(cu_res)
        assert _res == [9]


@pytest.mark.torch
def test_callback_events_torch():
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    from fastNLP.core.controllers.trainer import Trainer
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
    from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)
    optimizer = SGD(model.parameters(), lr=0.0001)
    dataset = TorchNormalDataset_Classification(num_labels=3, feature_dimension=10)
    dataloader = DataLoader(dataset=dataset, batch_size=4)

    trainer = Trainer(model=model, driver="torch", device="cpu", train_dataloader=dataloader, optimizers=optimizer)
    def filter_fn(filter, trainer):
        if trainer.__heihei_test__ == 10:
            return True
        return False

    event_state = Event.on_train_begin(filter_fn=filter_fn)

    @Filter(filter_fn=event_state.filter_fn)
    def _fn(trainer, data):
        return data

    _res = []
    for i in range(100):
        trainer.__heihei_test__ = i
        cu_res = _fn(trainer, i)
        if cu_res is not None:
            _res.append(cu_res)
    assert _res == [10]










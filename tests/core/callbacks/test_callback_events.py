import pytest
from functools import reduce

from fastNLP.core.callbacks.callback_events import Filter


class TestFilter:

    def test_params_check(self):
        # 顺利通过
        _filter1 = Filter(every=10)
        _filter2 = Filter(once=10)
        _filter3 = Filter(filter_fn=lambda: None)

        # 触发 ValueError
        with pytest.raises(ValueError) as e:
            _filter4 = Filter()
        exec_msg = e.value.args[0]
        assert exec_msg == "If you mean your decorated function should be called every time, you do not need this filter."

        # 触发 ValueError
        with pytest.raises(ValueError) as e:
            _filter5 = Filter(every=10, once=10)
        exec_msg = e.value.args[0]
        assert exec_msg == "These three values should be only set one."

        # 触发 TypeError
        with pytest.raises(ValueError) as e:
            _filter6 = Filter(every="heihei")
        exec_msg = e.value.args[0]
        assert exec_msg == "Argument every should be integer and greater than zero"

        # 触发 TypeError
        with pytest.raises(ValueError) as e:
            _filter7 = Filter(once="heihei")
        exec_msg = e.value.args[0]
        assert exec_msg == "Argument once should be integer and positive"

        # 触发 TypeError
        with pytest.raises(TypeError) as e:
            _filter7 = Filter(filter_fn="heihei")
        exec_msg = e.value.args[0]
        assert exec_msg == "Argument event_filter should be a callable"

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

    def test_filter_fn(self):
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



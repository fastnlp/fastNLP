import pytest
from fastNLP import Trainer, Event


def test_on():
    with pytest.raises(TypeError):
        @Trainer.on(Event.on_before_backward())
        def before_backend():
            pass

    @Trainer.on(Event.on_before_backward())
    def before_backend(*args):
        pass

    with pytest.raises(TypeError):
        @Trainer.on(Event.on_before_backward())
        def before_backend(*args, s):
            pass

    @Trainer.on(Event.on_before_backward())
    def before_backend(*args, s=2):
        pass
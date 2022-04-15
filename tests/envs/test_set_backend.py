import os

from fastNLP.envs.set_backend import dump_fastnlp_backend
from tests.helpers.utils import Capturing
from fastNLP.core import synchronize_safe_rm


def test_dump_fastnlp_envs():
    filepath = None
    try:
        with Capturing() as output:
            dump_fastnlp_backend()
        filepath = os.path.join(os.path.expanduser('~'), '.fastNLP', 'envs', os.environ['CONDA_DEFAULT_ENV']+'.json')
        assert filepath in output[0]
        assert os.path.exists(filepath)
    finally:
        synchronize_safe_rm(filepath)

import pytest
from fastNLP.envs.imports import _module_available
from fastNLP.core.utils import f_tqdm_progress, f_rich_progress

def test_raise():
    if not _module_available('tqdm') or f_rich_progress.dummy or f_tqdm_progress.dummy:
        pytest.skip('No tqdm')
    t = f_rich_progress.add_task('test', total=10)
    with pytest.raises(AssertionError):
        f_tqdm_progress.add_task('test')

    f_rich_progress.destroy_task(t)

    t = f_tqdm_progress.add_task('test', total=10)
    with pytest.raises(AssertionError):
        f_rich_progress.add_task('test')
__all__ = [
    're_instantiate_sampler',
    'conversion_between_reproducible_and_unrepeated_sampler'
]

from fastNLP.core.samplers.unrepeated_sampler import *
from fastNLP.core.samplers.reproducible_sampler import *


def conversion_between_reproducible_and_unrepeated_sampler(sampler):
    """
    将 sampler 替换成其对应的 reproducible 版本或 unrepeated 版本。如果输入是 UnrepeatedSampler 但是没找到对应的
        ReproducibleSampler，

    :param sampler:
    :return:
    """
    assert isinstance(sampler, UnrepeatedSampler) or isinstance(sampler, ReproducibleSampler), \
        "The sampler must be UnrepeatedSampler or ReproducibleSampler"
    if isinstance(sampler, UnrepeatedSampler):
        if isinstance(sampler, UnrepeatedRandomSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=RandomSampler)
        elif isinstance(sampler, UnrepeatedSequentialSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=SequentialSampler)
        elif isinstance(sampler, UnrepeatedSortedSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=SortedSampler)
        raise TypeError(f"{sampler.__class__} has no unrepeated version.")
    else:
        if isinstance(sampler, RandomSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=UnrepeatedRandomSampler)
        elif isinstance(sampler, SequentialSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=UnrepeatedSequentialSampler)
        elif isinstance(sampler, SortedSampler):
            return re_instantiate_sampler(sampler, new_sampler_class=UnrepeatedSortedSampler)
        raise TypeError(f"{sampler.__class__} has no reproducible version.")


def re_instantiate_sampler(sampler, new_sampler_class=None):
    all_attributes = vars(sampler)
    if new_sampler_class is not None:
        return new_sampler_class(**all_attributes)
    return type(sampler)(**all_attributes)
import numpy
import torch
import random


def set_rng_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # print('RNG_SEED {}'.format(seed))

import torch
import math

from torch import Tensor

DATASET_SIZE = 1000

def generate_disc_set(nb):
    inputs = Tensor(nb, 2).uniform_(0, 1)
    targets = inputs.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return inputs, targets

def gen_train_test():
    train_input, train_target = generate_disc_set(DATASET_SIZE)
    test_input, test_target = generate_disc_set(DATASET_SIZE)
    return train_input, train_target, test_input, test_target
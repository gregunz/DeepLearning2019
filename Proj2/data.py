import torch
import math

from torch import Tensor

def generate_disc_set(nb):
    inputs = Tensor(nb, 2).uniform_(0, 1)
    targets = inputs.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    return inputs, targets
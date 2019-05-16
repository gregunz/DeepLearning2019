import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import gen_train_test, transform_target
from deepy.loss import MSE
from deepy.nn import Linear, Sequential, Relu, Tanh
from deepy.optim import SGD
from deepy.tensor import Variable
from deepy.utils import graph_repr, accuracy

BATCH_SIZE = 100
NUM_ROUNDS = 1
NUM_EPOCHS = 1000
SEED = int(time.time())

torch.manual_seed(SEED)

print_once = True
losses_per_round = []
accuracies = {'train': [], 'test': []}


def init_model():
    return Sequential([
        Linear(2, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25, 2),
        Tanh()
    ])


print(graph_repr(init_model(), gen_train_test()[0][0]))

for _ in tqdm(range(NUM_ROUNDS)):  # rounds

    train_input, train_target, test_input, test_target = gen_train_test()
    # change the label to be -1 and 1 because the output of tanh is [-1,1]
    train_target, test_target = transform_target(train_target), transform_target(test_target)
    dataset_size = train_input.shape[0]

    # define the network
    model = init_model()
    optimizer = SGD(model.parameters(), lr=0.1 / BATCH_SIZE)  # , weight_decay=0.01)
    criterion = MSE()

    loss_per_epoch = []

    for _ in range(NUM_EPOCHS):  #  epochs$

        loss_per_batch = []
        for batch_idx in range(0, dataset_size, BATCH_SIZE):  #  batches
            batch_train_input = train_input[np.random.permutation(dataset_size)][batch_idx:batch_idx + BATCH_SIZE]
            batch_train_target = train_target[np.random.permutation(dataset_size)][batch_idx:batch_idx + BATCH_SIZE]

            optimizer.zero_grad()

            # we train with batches of the size of the dataset
            x = Variable(batch_train_input)
            out = model(x)
            loss = criterion(out, batch_train_target)
            loss_per_batch.append(loss.data)
            loss.backward()
            optimizer.step()
        loss_per_epoch.append(np.mean(loss_per_batch))

    losses_per_round.append(loss_per_epoch)
    accuracies['train'].append(accuracy(model(train_input).data.argmax(1), train_target.argmax(1)))
    accuracies['test'].append(accuracy(model(test_input).data.argmax(1), test_target.argmax(1)))

loss_plot_data = np.mean(losses_per_round, axis=0)
plt.plot(loss_plot_data)
plt.savefig('tmp.png')

print(f"Mean accuracy on train set: {np.mean(accuracies['train']):.4f} with std = {np.std(accuracies['train']):.4f}")
print(f"Mean accuracy on test set: {np.mean(accuracies['test']):.4f} with std = {np.std(accuracies['test']):.4f}")

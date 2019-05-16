import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import gen_train_test, transform_target
from deepy.loss import MSE
from deepy.nn import Linear, Sequential, Relu, Tanh
from deepy.optim import SGD
from deepy.tensor import Variable
from deepy.utils import graph_repr, accuracy

BATCH_SIZE = 32
SEED = 42

torch.manual_seed(SEED)

train_input, train_target, test_input, test_target = gen_train_test()
dataset_size = train_input.shape[0]

# define the network
model = Sequential([
    Linear(2, 25),
    Relu(),
    Linear(25, 25),
    Relu(),
    Linear(25, 2),
    Tanh()
])

print(graph_repr(model, train_input[0]))

# change the label to be -1 and 1 because the output of tanh is [-1,1]
train_target, test_target = (transform_target(train_target), transform_target(test_target))

optimizer = SGD(model.parameters(), lr=0.01 / BATCH_SIZE, weight_decay=0.01)
criterion = MSE()

losses = []
for i in tqdm(range(100)):
    for batch_idx in range(0, dataset_size, BATCH_SIZE):
        batch_train_input = train_input[batch_idx:batch_idx + BATCH_SIZE]
        batch_train_target = train_target[batch_idx:batch_idx + BATCH_SIZE]
        optimizer.zero_grad()

        # we train with batches of the size of the dataset
        x = Variable(batch_train_input)
        out = model(x)
        loss = criterion(out, batch_train_target)
        losses.append(loss.data)
        loss.backward()
        optimizer.step()

plt.plot(losses)
plt.savefig('tmp.png')

pred = model(train_input)
print(f"Accuracy on train set: {accuracy(pred.data, train_target.argmax(1)):.4f}")

pred = model(test_input)
print(f"Accuracy on test set:  {accuracy(pred.data, test_target.argmax(1)):.4f}")

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from data import gen_train_test, transform_target, generate_disc_set, DATASET_SIZE
from deepy.loss import MSE
from deepy.nn import Linear, Sequential, Relu, Tanh
from deepy.optim import SGD
from deepy.tensor import Variable
from deepy.utils import graph_repr, accuracy

BATCH_SIZE = 40
NUM_ROUNDS = 10
NUM_EPOCHS = 500
SEED = 42

torch.manual_seed(SEED)

best_test_acc = 0
train_loss = torch.empty((NUM_ROUNDS, NUM_EPOCHS))
accuracies = {'train': torch.empty(NUM_ROUNDS), 'test': torch.empty(NUM_ROUNDS)}


def init_data():
    train_input, train_target, test_input, test_target = gen_train_test()
    # change the label to be -1 and 1 because the output of tanh is [-1,1]
    train_target, test_target = transform_target(train_target), transform_target(test_target)
    return train_input, train_target, test_input, test_target


def init_model():
    return Sequential([
        Linear(2, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25, 2),
        Tanh()
    ])


best_model = init_model()
print(graph_repr(best_model, init_data()[0][0]))

for round_idx in tqdm(range(NUM_ROUNDS)):  # rounds

    # Loading data (randomly)
    train_input, train_target, test_input, test_target = init_data()

    # Initializing network
    model = init_model()
    optimizer = SGD(model.parameters(), lr=0.005)
    criterion = MSE()

    for epoch_idx in range(NUM_EPOCHS):  #  epochs

        epoch_loss = 0
        permutations = torch.randperm(DATASET_SIZE)
        for batch_idx in range(0, DATASET_SIZE, BATCH_SIZE):  #  batches
            # we train with batches
            batch_train_input = train_input[permutations][batch_idx:batch_idx + BATCH_SIZE]
            batch_train_target = train_target[permutations][batch_idx:batch_idx + BATCH_SIZE]

            optimizer.zero_grad()
            x = Variable(batch_train_input)
            out = model(x)
            loss = criterion(out, batch_train_target)
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

        train_loss[round_idx, epoch_idx] = epoch_loss * BATCH_SIZE / DATASET_SIZE

    train_acc = accuracy(model(train_input).data.argmax(1), train_target.argmax(1))
    accuracies['train'][round_idx] = train_acc
    test_acc = accuracy(model(test_input).data.argmax(1), test_target.argmax(1))
    accuracies['test'][round_idx] = test_acc
    if test_acc > best_test_acc:
        best_model = model

# Printing accuracy estimates
print(f"Train accuracy (mean on {NUM_ROUNDS} rounds): {accuracies['train'].mean():.4f} with std = {accuracies[
    'train'].std():.4f}")
print(f"Test accuracy (mean on {NUM_ROUNDS} rounds):  {accuracies['test'].mean():.4f} with std = {accuracies[
    'test'].std():.4f}")

# Plotting training loss
loss_plot_data = train_loss.mean(0)
plt.title('Loss (train)')
plt.plot(loss_plot_data.numpy())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()

# Evaluation on newly generated data to create a plot
validation_input, validation_target = generate_disc_set(DATASET_SIZE)
sns.scatterplot(
    x=validation_input.numpy()[:, 0],
    y=validation_input.numpy()[:, 1],
    hue=best_model(validation_input).data.argmax(1).numpy()
)
plt.savefig('plot.png')
plt.show()

import torch

from matplotlib import pyplot as plt

from constants import NUM_EPOCHS, NUM_ROUNDS, SEED
from evaluation import evaluate_model
from models import SimpleNN, SimpleCNN

torch.manual_seed(SEED)


def print_training(model_name, with_aux):
    s = '' if with_aux else 'out'
    print(f'Training "{model_name}" model with{s} auxiliary training loss')


# Model evaluation
print_training('Fully Connected (FC)', with_aux=False)
loss_fc, acc_fc = evaluate_model(SimpleNN, num_epochs=NUM_EPOCHS, num_rounds=NUM_ROUNDS)

print_training('Fully Connected (FC_aux)', with_aux=True)
loss_fc_aux, acc_fc_aux = evaluate_model(SimpleNN, num_epochs=NUM_EPOCHS, num_rounds=NUM_ROUNDS, with_aux_classes=True)

print_training('Convolutional (CNN)', with_aux=False)
loss_cnn, acc_cnn = evaluate_model(SimpleCNN, num_epochs=NUM_EPOCHS, num_rounds=NUM_ROUNDS)

print_training('Convolutional (CNN_aux)', with_aux=True)
loss_cnn_aux, acc_cnn_aux = \
    evaluate_model(SimpleCNN, num_epochs=NUM_EPOCHS, num_rounds=NUM_ROUNDS, with_aux_classes=True)

# Statistics & Plots
accuracies = [acc_fc, acc_fc_aux, acc_cnn, acc_cnn_aux]
losses = [loss_fc, loss_fc_aux, loss_cnn, loss_cnn_aux]
names = ['FC', 'FC_aux', 'CNN', 'CNN_aux']


def plot(title, filename, tensor):
    plt.figure(figsize=(16, 9))
    plt.title(title)
    plt.plot(tensor.transpose(1, 0).numpy())
    plt.legend(names)
    plt.savefig(filename)


# Plotting accuracy during the test phase
plot(
    'Accuracy (test)',
    'accuracy_test.png',
    torch.cat([torch.mean(acc['test'], dim=0, keepdim=True) for acc in accuracies])
)

# Plotting loss during the train phase
plot(
    'Loss (train phase)',
    'loss_train.png',
    torch.cat([torch.mean(loss['train'], dim=0, keepdim=True) for loss in losses])
)

# Plotting loss during the test phase
plot(
    'Loss (test phase)',
    'loss_test.png',
    torch.cat([torch.mean(loss['test'], dim=0, keepdim=True) for loss in losses])
)

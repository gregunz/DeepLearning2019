import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as utils

from matplotlib import pyplot as plt

import dlc_practical_prologue as dlc

from train import train_model
from constants import N

criterion = nn.BCELoss()

def to_dataloader(X, Y, batch_size=32):
    dset = utils.TensorDataset(X, Y) # create your datset
    return utils.DataLoader(dset, batch_size=batch_size, shuffle=True) # create your dataloader

def evaluate_model(model_callable, num_epochs=20, num_rounds=10, with_plot=True):
    model = model_callable()
    model_optim = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    train_losses = torch.empty([num_rounds, num_epochs])
    losses = {'train':train_losses, 'val':torch.empty_like(train_losses)}
    accuracies = {'train': torch.empty_like(train_losses), 'val': torch.empty_like(train_losses)}
    
    for round_idx in range(num_rounds):
        train_input, train_target, train_classes,\
        test_input, test_target, test_classes = dlc.generate_pair_sets(N)
        dataloaders = {'train': to_dataloader(train_input, train_target),'val': to_dataloader(test_input, test_target) }
        dataset_sizes = {phase:len(data) for phase, data in dataloaders.items()}

        _, round_losses, round_accuracies = \
        train_model(dataloaders, dataset_sizes, model, 'cpu', criterion, model_optim, num_epochs=num_epochs)
        
        for phase in ['train', 'val']:
            losses[phase][round_idx] = torch.tensor(round_losses[phase])
            accuracies[phase][round_idx] = torch.tensor(round_accuracies[phase])

    mean_losses, mean_accuracies = dict(), dict()
    for phase in ['train', 'val']:
        mean_losses[phase] = torch.mean(losses[phase], dim=0).numpy()
        mean_accuracies[phase] = torch.mean(accuracies[phase], dim=0).numpy()

    if with_plot:
        # Plotting loss
        plt.title('Loss (mean over {} rounds)'.format(num_rounds))
        plt.xticks(list(range(num_epochs)))
        best_loss_idx = np.argmin(mean_losses['val'])
        plt.plot(mean_losses['train'])
        plt.axvline(x=best_loss_idx, color='red')
        plt.plot(mean_losses['val'])
        plt.show()

        # Plotting accuracy
        plt.title('Accuracy (mean over {} rounds)'.format(num_rounds))
        plt.xticks(list(range(num_epochs)))
        best_acc_idx = np.argmax(mean_accuracies['val'])
        plt.plot(mean_accuracies['train'])
        plt.axvline(x=best_acc_idx, color='red')
        plt.plot(mean_accuracies['val']);
        plt.show()
    
    best_accuracies = torch.max(accuracies['val'], dim=0)[0]
    print('Best validation accuracy (mean over {} rounds) = {:.4f}'.format(num_rounds, best_accuracies.mean()))
    print('and standard deviation = {:.4f}'.format(best_accuracies.std()))
    
    return losses, accuracies

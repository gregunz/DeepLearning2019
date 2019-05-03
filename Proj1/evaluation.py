import time
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as utils

from matplotlib import pyplot as plt

import dlc_practical_prologue as dlc

from train import train_model
from constants import N

def to_dataloader(X, Y, batch_size=32):
    dset = utils.TensorDataset(X, Y) # create your datset
    return utils.DataLoader(dset, batch_size=batch_size, shuffle=True) # create your dataloader

def evaluate_model(model_callable, num_epochs=20, num_rounds=10, with_plot=True, with_aux_classes=False):
    since = time.time()
    
    model = model_callable()
    print('Model parameters = {}'.format(sum(p.numel() for p in model.parameters())))
    model_optim = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()
    aux_criterion = None
    if with_aux_classes:
        aux_criterion = nn.CrossEntropyLoss()
        
    
    train_losses = torch.empty([num_rounds, num_epochs])
    losses = {'train':train_losses, 'val':torch.empty_like(train_losses)}
    accuracies = {'train': torch.empty_like(train_losses), 'val': torch.empty_like(train_losses)}
    
    for round_idx in range(num_rounds):
        train_input, train_target, train_classes,\
        test_input, test_target, test_classes = dlc.generate_pair_sets(N)
        if with_aux_classes:
            train_target = torch.cat([train_target.unsqueeze(1), train_classes], dim=1)
            test_target = torch.cat([test_target.unsqueeze(1), test_classes], dim=1)
        dataloaders = {'train': to_dataloader(train_input, train_target),'val': to_dataloader(test_input, test_target) }
        dataset_sizes = {phase:len(data) for phase, data in dataloaders.items()}

        _, round_losses, round_accuracies = \
        train_model(dataloaders, dataset_sizes, model, 'cpu', criterion, model_optim, num_epochs=num_epochs, aux_criterion=aux_criterion)
        
        for phase in ['train', 'val']:
            losses[phase][round_idx] = torch.tensor(round_losses[phase])
            accuracies[phase][round_idx] = torch.tensor(round_accuracies[phase])

    mean_losses, mean_accuracies = dict(), dict()
    for phase in ['train', 'val']:
        mean_losses[phase] = torch.mean(losses[phase], dim=0).numpy()
        mean_accuracies[phase] = torch.mean(accuracies[phase], dim=0).numpy()

    if with_plot:
        # Plotting loss
        plt.figure(figsize=(16,9))
        plt.title('Loss (mean over {} rounds)'.format(num_rounds))
        plt.xticks(list(range(num_epochs)))
        best_loss_idx = np.argmin(mean_losses['val'])
        plt.plot(mean_losses['train'])
        plt.axvline(x=best_loss_idx, color='red')
        plt.plot(mean_losses['val'])
        plt.show()

        # Plotting accuracy
        plt.figure(figsize=(16,9))
        plt.title('Accuracy (mean over {} rounds)'.format(num_rounds))
        plt.xticks(list(range(num_epochs)))
        best_acc_idx = np.argmax(mean_accuracies['val'])
        plt.plot(mean_accuracies['train'])
        plt.axvline(x=best_acc_idx, color='red')
        plt.plot(mean_accuracies['val']);
        plt.show()
    
    mean_accuracies = torch.mean(accuracies['val'], dim=0)
    print('Best validation accuracy (mean over {} rounds) = {:.4f}'.format(num_rounds, mean_accuracies.max()))
    #print('and standard deviation = {:.4f}'.format())
    
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return losses, accuracies

import copy
import sys
import time

import torch

from constants import AUX_LOSS_FACTOR


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, scheduler=None, writer=None, num_epochs=25,
                verbose=False, aux_criterion=None, aux_loss_factor=AUX_LOSS_FACTOR):
    """
    Train a model a number of epoch and keep weights of the best performing one (smallest validation loss)
    :param dataloaders: dict, dataloader for each phase
    :param dataset_sizes: dict, size of the dataset for each phase
    :param model: nn.Module, the model
    :param criterion: function, criterion to compute the loss
    :param optimizer: optimizer
    :param scheduler: Scheduler
    :param writer: tensorboard.Writer
    :param num_epochs: int, number of epoch
    :param verbose: bool, whether to print details such as loss or training duration
    :param aux_criterion: function, criterion to compute the auxilary loss
    :param aux_loss_factor: float, factor associated with the auxiliary loss
    :return: tuple(model, torch.Tensor, torch.Tensor), the model trained and the loss and the accuracy
    """
    since = time.time()

    epoch_losses = {'train': [], 'test': []}
    epoch_accuracies = {'train': [], 'test': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = sys.maxsize

    try:
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    if scheduler:
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_acc = 0.0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    if aux_criterion is not None:
                        aux_labels = labels[:, 1:]
                        labels = labels[:, 0]
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, aux_outputs = model(inputs)
                        loss = criterion(outputs, labels.float())
                        # in order to be able to compare models more easily
                        loss_true = loss.clone().item()
                        if aux_criterion is not None:
                            loss += aux_criterion(aux_outputs[:, :10], aux_labels[:, 0]) * aux_loss_factor
                            loss += aux_criterion(aux_outputs[:, 10:], aux_labels[:, 1]) * aux_loss_factor

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    loss_scalar = loss_true * inputs.size(0)
                    running_loss += loss_scalar
                    preds = (outputs > 0.5).long()
                    running_acc += torch.sum(preds == labels.data).double() / preds.nelement()

                    # tensorboard
                    if writer is not None:
                        x_axis = i + epoch * dataset_sizes[phase]
                        writer.add_scalar('{}_loss'.format(phase), loss_scalar, x_axis)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_losses[phase].append(epoch_loss)
                epoch_acc = running_acc / dataset_sizes[phase]
                epoch_accuracies[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'test' and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())


    except KeyboardInterrupt:
        pass

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Loss: {:.4f}'.format(best_loss))
        print('Best test Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses, epoch_accuracies

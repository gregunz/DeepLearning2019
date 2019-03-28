import time
import sys
import copy
import torch

def train_model(dataloaders, dataset_sizes, model, device, criterion, optimizer, scheduler=None, writer=None, num_epochs=25):
    since = time.time()
    
    epoch_losses = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = sys.maxsize

    try:
        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    if scheduler:
                        scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_acc  = 0.0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.float())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    
                    # statistics
                    loss_scalar =  loss.item() * inputs.size(0)
                    running_loss += loss_scalar
                    preds = (outputs > 0.5).long()
                    running_acc += torch.sum(preds == labels.data).double() / preds.nelement()

                    # tensorboard
                    if writer :
                        x_axis = i + epoch * dataset_sizes[phase]
                        writer.add_scalar('{}_loss'.format(phase), loss_scalar,  x_axis)
                    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_losses[phase].append(epoch_loss)
                epoch_acc = running_acc / dataset_sizes[phase]

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())


    except KeyboardInterrupt:
        pass
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_losses
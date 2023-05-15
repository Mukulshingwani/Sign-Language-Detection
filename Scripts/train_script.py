import torch
import torchvision
import livelossplot
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from livelossplot import PlotLosses
import copy
import time

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=10):
    """_summary_
    The function train_model trains a given model using the provided dataloaders and criterion for a specified number of epochs, and returns the trained model along with various training statistics. The function also uses a learning rate scheduler to adjust the learning rate during training and saves the best model weights based on top-5 accuracy achieved on the validation set.

    The function takes the following arguments:
    model: a PyTorch model to be trained
    dataloaders: a dictionary containing PyTorch dataloaders for the training and validation sets
    dataset_sizes: a dictionary containing the size of the training and validation datasets
    criterion: the loss function to be used for optimization
    optimizer: the optimization algorithm to be used for training
    scheduler: the learning rate scheduler to be used during training
    device: the device to be used for training (e.g. 'cpu' or 'cuda')
    num_epochs: the number of epochs to train the model (default: 10)
    
    The function returns the following:
    model: the trained PyTorch model
    train_loss: a list of training losses at each epoch
    test_loss: a list of validation losses at each epoch
    train_acc_top1: a list of top-1 training accuracies at each epoch
    test_acc_top1: a list of top-1 validation accuracies at each epoch
    train_acc_top5: a list of top-5 training accuracies at each epoch
    test_acc_top5: a list of top-5 validation accuracies at each epoch
    """
    model = model.to(device)
    since = time.time()
    liveloss = PlotLosses()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_top5 = 0.0

    train_loss = []
    test_loss = []
    train_acc_top1 = []
    test_acc_top1 = []
    train_acc_top5 = []
    test_acc_top5 = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_top1 = 0
            running_corrects_top5 = 0

            # Iterate over data.
            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.topk(outputs, k=5, dim=1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects_top1 += torch.sum(preds[:, 0] == labels)
                running_corrects_top5 += torch.sum(preds == labels.unsqueeze(1))
                print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")

                #print( (i+1)*100. / len(dataloaders[phase]), "% Complete" )
                # sys.stdout.flush()
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_top1 = running_corrects_top1.double() / dataset_sizes[phase]
            epoch_acc_top5 = running_corrects_top5.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc_top1 = epoch_acc_top1
                t_acc_top5 = epoch_acc_top5
            else:
                val_loss = epoch_loss
                val_acc_top1 = epoch_acc_top1
                val_acc_top5 = epoch_acc_top5
            
            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc_top5 > best_acc_top5:
                best_acc_top5 = epoch_acc_top5
                best_model_wts = copy.deepcopy(model.state_dict())

        train_loss.append(avg_loss)
        test_loss.append(val_loss)
        train_acc_top1.append(t_acc_top1.item())
        test_acc_top1.append(val_acc_top1.item())
        train_acc_top5.append(t_acc_top5.item())
        test_acc_top5.append(val_acc_top5.item())

        logs = {
            'loss': avg_loss,
            'val_loss': val_loss,
            'Top-1 accuracy': t_acc_top1.item(),
            'val_Top-1 accuracy': val_acc_top1.item(),
            'Top-5 accuracy': t_acc_top5.item(),
            'val_Top-5 accuracy': val_acc_top5.item()
        }
        liveloss.update(logs)
        liveloss.send()

        print('Train Loss: {:.4f} Top-1 Acc: {:.4f} Top-5 Acc: {:.4f}'.format(avg_loss, t_acc_top1, t_acc_top5))
        print(  'Val Loss: {:.4f} Top-1 Acc: {:.4f} Top-5 Acc: {:.4f}'.format(val_loss, val_acc_top1, val_acc_top5))
        print('Best Val Top-5 Accuracy: {}'.format(best_acc_top5))
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Top-5 Acc: {:4f}'.format(best_acc_top5))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, test_loss, train_acc_top1, test_acc_top1, train_acc_top5, test_acc_top5 
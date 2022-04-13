import time
import copy
import torch
from tqdm import tqdm
import wandb

wandb.init(project="Digikala-Watermark-Detection", entity="mohammad-jafari")

wandb.config = {
    "learning_rate": 0.001,
    "epochs": 20,
    "batch_size": 256
}


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            epoch_all = 0
            epoch_loss = 0

            # Iterate over data.
            with tqdm(enumerate(dataloaders[phase]), unit="batch", total=len(dataloaders[phase])) as tepoch:
                for i, (inputs, labels) in tepoch:
                    inputs = inputs.float()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    outputs = None
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.

                        outputs = model(inputs).float()
                        loss = criterion(outputs.float(), labels.unsqueeze(1).float())

                        preds = torch.round(outputs)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_loss += float(loss)
                    epoch_all += len(outputs)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.count_nonzero(preds == labels.unsqueeze(1))
                    tepoch.set_description(
                        f'{phase} - Loss: {epoch_loss / (i + 1):.3e} - Acc: {running_corrects * 100. / epoch_all:.2f}%')

                wandb.log(
                    {f"{phase} loss": epoch_loss / (i + 1), f'{phase} Accuracy': running_corrects * 100. / epoch_all})
                wandb.watch(model)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'model_weights-2.pt')
    return model, val_acc_history

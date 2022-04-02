import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy as np
import os
from dataset import WatermarkDataset
from matplotlib import pyplot as plt
from transformers import train_transforms_simple, train_transforms
from model import initialize_model
from train import train_model
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
current_path = os.getcwd()
model_name = "resnet"
num_classes = 2
batch_size = 64
num_epochs = 15
train_val_rate = 0.75
feature_extract = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device is: {device}')
AUGMENTATION_RATE = 1

train_data_path = os.path.join(current_path, 'dataset\\train\\')
print(f'Train data path is: {train_data_path}')
classes = [0, 1]  # 0 -> Negative | 1 -> Positive
labels = []
image_paths = []
train_image_paths = []

def show(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    return cv2.cvtColor(np.transpose(np_img, (1, 2, 0)), cv2.COLOR_RGB2BGR)



for data_path in glob.glob(train_data_path + '\\*'):
    train_image_paths.append(glob.glob(data_path + '\\*'))

for i, class_samples in enumerate(train_image_paths):
    for sample in class_samples:
        labels.append(classes[i])
        image_paths.append(sample)

main_dataset = WatermarkDataset(image_paths, labels, train_transforms_simple)
# data_loader = DataLoader(main_dataset, batch_size=batch_size)
# print(data_loader)
# print(len(data_loader.dataset))
# batch = next(iter(data_loader))
# grid = torchvision.utils.make_grid(batch[0], nrow=8, padding=30)
# grid_img = show(grid)
# cv2.imwrite('grid-original.jpg', grid_img)

for i in range(AUGMENTATION_RATE):
    augmented_dataset = WatermarkDataset(image_paths, labels, train_transforms)
    main_dataset = ConcatDataset([main_dataset, augmented_dataset])
    # data_loader = DataLoader(main_dataset, batch_size=batch_size, shuffle=True)
    # batch = next(iter(data_loader))
    # grid = torchvision.utils.make_grid(batch[0], nrow=8, padding=30)
    # grid_img = show(grid)
    # cv2.imwrite(f'grid-augmented-{i}.jpg', grid_img)


indices = np.random.choice(np.arange(len(main_dataset)), len(main_dataset), replace=False)
train_indices, val_indices = indices[:int(train_val_rate * len(main_dataset))], indices[int(train_val_rate * len(main_dataset)):]
train_set, val_set = Subset(main_dataset, train_indices), Subset(main_dataset, val_indices)

print(f'Train dataset length is: {len(train_set)}')
print(f'Val dataset length is: {len(val_set)}')

dataloaders_dict = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True)}
print(dataloaders_dict)
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft.to(device)
print(model_ft)

params_to_update = model_ft.parameters()

print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs,
                             is_inception=(model_name == "inception"))

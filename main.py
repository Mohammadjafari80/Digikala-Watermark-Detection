import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
import numpy as np
import random
import torchvision
from dataset import WatermarkDataset
from matplotlib import pyplot as plt
from transformers import train_transforms_simple, train_transforms


# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

def show(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation='nearest')
    return cv2.cvtColor(np.transpose(np_img, (1, 2, 0)), cv2.COLOR_RGB2BGR)


train_data_path = '.\\dataset\\train\\'
classes = [0, 1]  # 0 -> Negative | 1 -> Positive
labels = []
image_paths = []
train_image_paths = []

for data_path in glob.glob(train_data_path + '\\*'):
    train_image_paths.append(glob.glob(data_path + '\\*'))

for i, class_samples in enumerate(train_image_paths):
    for sample in class_samples:
        labels.append(classes[i])
        image_paths.append(sample)

main_dataset = WatermarkDataset(image_paths, labels, train_transforms_simple)
data_loader = DataLoader(main_dataset, batch_size=64)
batch = next(iter(data_loader))
grid = torchvision.utils.make_grid(batch[0], nrow=8, padding=30)
grid_img = show(grid)
cv2.imwrite('grid.jpg', grid_img)

for i in range(3):
    augmented_dataset = WatermarkDataset(image_paths, labels, train_transforms)
    main_dataset = ConcatDataset([main_dataset, augmented_dataset])
    data_loader = DataLoader(augmented_dataset, batch_size=64)
    batch = next(iter(data_loader))
    grid = torchvision.utils.make_grid(batch[0], nrow=8, padding=30)
    grid_img = show(grid)
    cv2.imwrite(f'grid-{i}.jpg', grid_img)

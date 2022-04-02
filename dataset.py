from torch.utils.data import Dataset
import cv2


class WatermarkDataset(Dataset):
    def __init__(self, image_paths, labels, transform=False):
        self.labels = labels
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

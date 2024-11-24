import pandas as pd
from PIL import Image
from PIL.ImageOps import grayscale
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import yolo


class ImageNetteDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.split = (split != 'train')
        self.root = root
        self.transform = transform
        self.images = pd.read_csv(
            root + '/noisy_imagenette.csv'
        )[['path', 'noisy_labels_0', 'is_valid']]
        self.images = self.images[self.images['is_valid'] == self.split]
        self.images['noisy_labels_0'] = pd.Categorical(
            self.images['noisy_labels_0']).codes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images.iloc[idx]
        img = Image.open(self.root + '/' + item['path']).convert('RGB')
        mask = yolo.generate_bounding_boxes_with_white_region(self.root + '/' + item['path'])
        mask = Image.fromarray(mask).convert('L')
        mask = mask.resize(img.size)

        if self.transform:
            img = self.transform(img)
            # mask = self.transform(mask)
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor
            # mask = mask.repeat(3, 1, 1)  # Duplicate channels to make it 3-channel
            mask = transforms.Resize((img.shape[1], img.shape[2]))(mask)

        img_with_mask = torch.cat((img, mask[:1]), dim=0)

        return img_with_mask, item['noisy_labels_0']

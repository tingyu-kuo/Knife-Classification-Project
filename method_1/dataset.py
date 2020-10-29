from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.x1 = data[0]
        self.x2 = data[1]
        self.y = data[2]
        self.transform = transform

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        image1 = Image.open(self.x1[index]).convert('RGB')
        image2 = Image.open(self.x2[index]).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, self.y[index]


def data(base):
    parent_path = os.path.abspath("..")
    data_root = os.path.join(parent_path, 'WearDepthMap_0308')
    data_root = Path(data_root)
    if base == 'broken':
        classes = ['broken', 'rough', 'pre', 'new']
    elif base == 'new':
        classes = ['new', 'pre', 'rough', 'broken']
    img_1, img_2, label = [], [], []

    for i, broken_img in enumerate(os.listdir(data_root/base)):
        for j, class_name in enumerate(classes):
            folder_root = data_root / class_name
            all_img = os.listdir(folder_root)
            if class_name == base:
                for k in range(i+1, len(all_img)):
                    img_1.append(data_root/base/broken_img)
                    img_2.append(data_root/base/all_img[k])
                    label.append(j)
            else:
                for k in range(len(all_img)):
                    img_1.append(data_root/base/broken_img)
                    img_2.append(data_root/class_name/all_img[k])
                    label.append(j)

    return img_1, img_2, label


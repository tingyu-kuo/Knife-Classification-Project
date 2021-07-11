import torch
from pathlib import Path
import torch.nn as nn
from PIL import Image
import pandas as pd
from torchvision.transforms.transforms import RandomResizedCrop
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau



class MyModel(nn.Module):
    def __init__(self, num_classes=None):
        super(MyModel, self).__init__()
        model = models.resnet50(pretrained=True)
        self.extract = nn.Sequential(*(list(model.children())[:-4]))
        self.norm = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.Dropout2d(0.5)
        )

        self.classifer = nn.Sequential(
            *(list(model.children())[-4:-1]),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes)     
        )

    def forward(self, x1, x2):
        x1 = self.extract(x1)
        x2 = self.extract(x2)
        diff = torch.pow(x2 - x1, 2)
        diff = self.norm(diff)
        x2 = x2 * diff
        x = self.classifer(x2)
        return x

class TrainDataset(Dataset):
    def __init__(self, data_n=20, transform=None):
        self.x1, self.x2, self.y = [], [], []
        self.transform = transform
        data = pd.read_csv('train_final.csv')
        cls_0 = list((data[data['ClassId']==0])['ImageId'])
        cls_1 = list((data[data['ClassId']==1])['ImageId'])
        cls_2 = list((data[data['ClassId']==2])['ImageId'])
        cls_3 = list((data[data['ClassId']==3])['ImageId'])
        cls_4 = list((data[data['ClassId']==4])['ImageId'])
        f_cls = [cls_0, cls_1, cls_2, cls_3, cls_4]
        img_root = Path('train_images_crop')

        for i in range(data_n):
            for t, c in enumerate(f_cls):
                for j in range(i+1, data_n):
                    self.x1.append(img_root/f_cls[0][i])
                    self.x2.append(img_root/c[j])
                    self.y.append(t)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image1 = Image.open(self.x1[index]).convert('RGB')
        image2 = Image.open(self.x2[index]).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, self.y[index]


class TestDataset(Dataset):
    def __init__(self, data_n=20 , transform=None):
        self.x1, self.x2, self.y = [], [], []
        self.transform = transform
        data = pd.read_csv('train_final.csv')
        cls_0 = list((data[data['ClassId']==0])['ImageId'])
        cls_1 = list((data[data['ClassId']==1])['ImageId'])
        cls_2 = list((data[data['ClassId']==2])['ImageId'])
        cls_3 = list((data[data['ClassId']==3])['ImageId'])
        cls_4 = list((data[data['ClassId']==4])['ImageId'])
        f_cls = [cls_0, cls_1, cls_2, cls_3, cls_4]
        img_root = Path('train_images_crop')

        for i in range(data_n):
            for t, c in enumerate(f_cls):
                bound = data_n + int(data_n/5)
                upper_bound = len(cls_0)
                bound = bound if bound<= upper_bound else upper_bound
                for j in range(data_n, bound):
                    self.x1.append(img_root/f_cls[0][i])
                    self.x2.append(img_root/c[j])
                    self.y.append(t)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        image1 = Image.open(self.x1[index]).convert('RGB')
        image2 = Image.open(self.x2[index]).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, self.y[index]


def visualization(train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(f'{title}-CrossRef-50.png')

def train(epochs=50, lr=1e-3, batch_size=16):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    # model
    model = MyModel(num_classes=5)
    model = model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomResizedCrop(256),
        # transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    train_dataset = TrainDataset(transform=transform)
    test_dataset = TestDataset(transform=transform)
    # train_need = int(len(train_dataset)*0.5)
    # train_drop = len(train_dataset) - train_need
    # train_dataset, train_drop = random_split(train_dataset, [train_need, train_drop])
    # test_need = int(len(test_dataset)*0.5)
    # test_drop = len(test_dataset) - test_need
    # test_dataset, test_drop = random_split(test_dataset, [test_need, test_drop])
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=2, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        

        for inputs1, inputs2, labels in tqdm(train_loader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            outputs= model(inputs1, inputs2)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data).float()

        test_loss, test_acc = 0.0, 0.0
        for inputs1, inputs2, labels in tqdm(test_loader):
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)
            with torch.no_grad():
               outputs = model(inputs1, inputs2)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            test_loss += loss.item()
            test_acc += torch.sum(preds == labels.data).float()

        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc.to('cpu') / len(train_dataset)
        test_loss = test_loss / len(test_dataset)
        test_acc = test_acc.to('cpu') / len(test_dataset)

        # if train_loss < test_loss:
        #     model.fc1.requires_grad_ = False
        #     model.fc1.requires_grad_ = False

        scheduler.step(test_loss)

        train_loss_reg.append(train_loss)
        train_acc_reg.append(train_acc)
        test_loss_reg.append(test_loss)
        test_acc_reg.append(test_acc)

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {test_loss:.4f}\taccuracy: {test_acc:.4f}\n')
        # print(f'Train loss: {train_loss:.4f}')
        # print(f'Test loss: {test_loss:.4f}')
    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')
if __name__ == '__main__':
    train()

import pandas as pd
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.x, self.y = [], []
        self.transform = transform
        root = Path('.')
        ls = ['MT_Free', 'MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        for c in range(len(ls)):
            imgs = list((root / 'dataset' / ls[c]).glob('*.jpg'))
            imgs.sort()
            for img in imgs[:28]:
                self.x.append(img)
                self.y.append(c)
        print(len(self.x))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.x, self.y = [], []
        self.transform = transform
        root = Path('.')
        ls = ['MT_Free', 'MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        for c in range(len(ls)):
            imgs = list((root / 'dataset' / ls[c]).glob('*.jpg'))
            imgs.sort()
            for img in imgs[28:32]:
                self.x.append(img)
                self.y.append(c)
        print(len(self.x))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.y[index]

def visualization(train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(f'{title}-resnet50.png')


def train(epochs=30, lr=1e-4, batch_size=16):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # model
    model = models.resnet50(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 6)
    # model = torch.load('model_save/resnet50.pth')
    model = model.to(device)
  

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # train
    # data = SteelDataset(transform=transform)
    # train_len = int(len(data) * 0.85)
    # test_len = len(data) - train_len
    # train_dataset, test_dataset = random_split(data, [train_len, test_len])
    train_dataset = TrainDataset(transform=transform)
    test_dataset = TestDataset(transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data).float()

        model.eval()
        test_loss, test_acc = 0.0, 0.0
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            test_loss += loss.item()
            test_acc += torch.sum(preds == labels.data).float()

        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc.to('cpu') / len(train_dataset)
        test_loss = test_loss / len(test_dataset)
        test_acc = test_acc.to('cpu') / len(test_dataset)

        scheduler.step(test_loss)

        train_loss_reg.append(train_loss)
        train_acc_reg.append(train_acc)
        test_loss_reg.append(test_loss)
        test_acc_reg.append(test_acc)

        # if test_loss < best :
        #     best = test_loss
        #     torch.save(model, os.path.join('model_save', f'resnet50.pth'))

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {test_loss:.4f}\taccuracy: {test_acc:.4f}\n')
    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')

if __name__ == '__main__':
    train()
    # SteelDataset()
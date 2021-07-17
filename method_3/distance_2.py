import torch
import os
from pathlib import Path
import torch.nn as nn
from PIL import Image
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MyModel(nn.Module):
    def __init__(self, num_classes=None):
        super(MyModel, self).__init__()
        model = models.resnet50(pretrained=True)
        self.extract = nn.Sequential(
            *(list(model.children())[:-2]),
        )
        self.norm = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(inplace=True),
            nn.Dropout()
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=2048, out_features=2048),
        #     nn.BatchNorm1d(2048),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Dropout()
        # )

        self.classifer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes)
        )


    def forward(self, x1, x2):
        x1 = self.extract(x1)
        x2 = self.extract(x2)
        diff = torch.sqrt(torch.sum(torch.pow(x1-x2, 2), dim=[2,3]))
        diff = self.norm(diff)
        diff = nn.Flatten()(diff)
        # diff_ = self.fc(diff)
        # x = diff + diff_
        x = self.classifer(diff)
        return x

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.x1, self.x2, self.y = [], [], []
        self.transform = transform
        ls = ['MT_Free', 'MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        root = Path('.')
        base_imgs = list((root/'dataset'/ls[0]).glob('*.jpg'))
        base_imgs.sort()
        for i in range(len(base_imgs[:28])):
            for j in range(6):
                imgs = list((root/'dataset'/ls[j]).glob('*.jpg'))
                imgs.sort()
                for k in range(i+1, len(imgs[:28])):
                    self.x1.append(base_imgs[i])
                    self.x2.append(imgs[k])
                    self.y.append(j)

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
    def __init__(self, transform=None):
        self.x1, self.x2, self.y = [], [], []
        self.transform = transform
        ls = ['MT_Free', 'MT_Blowhole', 'MT_Break', 'MT_Crack', 'MT_Fray', 'MT_Uneven']
        root = Path('.')
        base_imgs = list((root/'dataset'/ls[0]).glob('*.jpg'))
        base_imgs.sort()
        for i in range(len(base_imgs[:28])):
            for j in range(6):
                imgs = list((root/'dataset'/ls[j]).glob('*.jpg'))
                imgs.sort()
                for k in range(28, 32):
                    self.x1.append(base_imgs[i])
                    self.x2.append(imgs[k])
                    self.y.append(j)

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
    plt.savefig(f'{title}-Distance-50-without-pool.png')

def train(epochs=30, lr=1e-4, batch_size=16):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # model
    model = MyModel(num_classes=6)
    model = model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomRotation(15),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TrainDataset(transform=transform)
    test_dataset = TestDataset(transform=transform)
    print(len(train_dataset))
    print(len(test_dataset))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params':model.parameters(), 'lr':lr}
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.6)
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
            outputs = model(inputs1, inputs2)
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

        scheduler.step(test_loss)

        train_loss_reg.append(train_loss)
        train_acc_reg.append(train_acc)
        test_loss_reg.append(test_loss)
        test_acc_reg.append(test_acc)

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {test_loss:.4f}\taccuracy: {test_acc:.4f}\n')
    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')
if __name__ == '__main__':
    train()
    # MyModel()

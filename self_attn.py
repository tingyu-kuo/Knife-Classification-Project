import torch
import os
from pathlib import Path
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from center_loss import CenterLoss


# Extract Features
class ExtractModel(nn.Module):
    def __init__(self, n_classes=None):
        super(ExtractModel, self).__init__()
        model = models.resnet18(pretrained=False)
        self.extract = nn.Sequential(*(list(model.children())[:-2]))

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        # self.act = nn.LeakyReLU(inplace=True)
        self.act = nn.Sigmoid()
        # self attention
        # query
        self.conv1x1_q = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.norm_q = nn.BatchNorm2d(512)
        # key
        self.conv1x1_k = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.norm_k = nn.BatchNorm2d(512)
        # value
        self.conv1x1_v = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.norm_v = nn.BatchNorm2d(512)
        # classifier
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            # nn.Dropout(0.3),
            nn.Flatten()
        )
        self.conv1x1_f = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        self.output = nn.Linear(in_features=512, out_features=n_classes)
        # self.gamma = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.extract(x)
        s = list(x.size())
        # query
        q = self.conv1x1_q(x)
        q = self.flatten(q)
        q = q.transpose(1,2)
        # key
        k = self.conv1x1_k(x)
        k = self.flatten(k)
        # value
        v = self.conv1x1_v(x)
        v = self.flatten(v)
        # attention map
        att_map = torch.bmm(q, k)
        att_map = att_map.transpose(1,2)
        att_map = nn.Softmax(dim=2)(att_map)
        # att_map = att_map.transpose(1,2)

        # feature map
        att = torch.bmm(v, att_map)
        att = torch.reshape(att, s)
        # att = self.conv1x1_f(att)
        # f = att*self.gamma + x
        f = self.avgpool(att)
        # classify
        # c = self.avgpool(x)
        c = self.output(f)
        return f, c

class TrainDataset(Dataset):
    def __init__(self, transform=None):
        self.x, self.y = [], []
        self.transform = transform
        root = Path('dataset_org')
        ls = ['1', '2', '3', '4']
        for c in range(len(ls)):
            imgs = list((root / ls[c]).glob('*.jpg'))
            for img in imgs[:160]:
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
        root = Path('dataset_org')
        ls = ['1', '2', '3', '4']
        for c in range(len(ls)):
            imgs = list((root / ls[c]).glob('*.jpg'))
            for img in imgs[160:190]:
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
    plt.savefig(f'curve/{title}-self_att_18.png')


def train(epochs=100, lr=1e-4, batch_size=16):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model
    model = ExtractModel(n_classes=4)
    model = model.to(device)


    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TrainDataset(transform=transform)
    test_dataset = TestDataset(transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    center_loss = CenterLoss(num_classes=4, feat_dim=512, use_gpu=device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW([
        {'params':model.parameters(), 'lr':lr},
        # {'params':center_loss.parameters(), 'lr':0.5}
    ])
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
            features, outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss_cls = criterion(outputs, labels)
            # loss_center = center_loss(features, labels)
            # loss = loss_center*0.01 + loss_cls

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
            with torch.no_grad():
                features, outputs = model(inputs)
            #     loss_cls = criterion(outputs, labels)
            #     loss_center = center_loss(features, labels)
            # loss = loss_center*0.01 + loss_cls
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
    # visualization(train_loss_reg, test_loss_reg, 'Loss')
    # visualization(train_acc_reg, test_acc_reg, 'Acc')

if __name__ == '__main__':
    train()
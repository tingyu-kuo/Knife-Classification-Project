import torch.nn as nn
import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset, ConcatDataset
from dataset import ImageDataset, data
from PIL import Image
import os
import torchvision.models as models
import datetime
from matplotlib import pyplot as plt


class Output_Layer(nn.Module):
    def __init__(self, num_classes=4):
        super(Output_Layer, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=num_classes),
        )
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        # x = x1-x2
        x = self.layers(x)
        return x


def date():
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    month = str('0' + month) if month < 10 else str(month)
    day = str('0' + day) if day < 10 else str(day)

    return month + day

def visualize(train_l, train_a, test_l, test_a, base):
    plt.figure()
    plt.plot(train_l, 'r', label="train loss")
    plt.plot(test_l, 'b', label="test loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{base}-loss-curve.png')

    plt.figure()
    plt.plot(train_a, 'r', label="train accuracy")
    plt.plot(test_a, 'b', label="test accuracy")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f'{base}-acc-curve.png')
    plt.show()

# parameters
LR = 0.001
BATCH_SIZE = 32
EPOCH = 100
CUDA = 3
SAVE_ROOT = os.path.join(os.getcwd(), 'model_pth')
DATE = date()
BASE = 'broken'

def train():

    model = models.resnet50(pretrained=True)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 1024)
    model = model.cuda(CUDA)
    model.train()

    output_layer = Output_Layer(num_classes=4).cuda(CUDA)
    output_layer.train()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_1, img_2, label = data(BASE)
    dataset = ImageDataset([img_1, img_2, label], transform=transform)
    train_len = int(len(dataset) * 0.75)
    test_len = int(len(dataset)) - train_len
    print(train_len, test_len)
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    optimizer2 = torch.optim.SGD(output_layer.parameters(), lr=LR, momentum=0.9)
    train_loss_curve, train_acc_curve = [], []
    test_loss_curve, test_acc_curve = [], []
    best = 10
    for epoch in range(EPOCH):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        print(f'\nEpoch: {epoch + 1}/{EPOCH}')
        print('-' * len(f'Epoch: {epoch + 1}/{EPOCH}'))

        for i, (input1, input2, labels) in enumerate(tqdm(train_data_loader)):
            input1 = input1.cuda(CUDA)
            input2 = input2.cuda(CUDA)
            labels = labels.long().cuda(CUDA)

            optimizer.zero_grad()
            optimizer2.zero_grad()

            output1 = model(input1)

            output2 = model(input2)

            output = output_layer(output1, output2)
            preds = torch.argmax(output, dim=1)

            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()
            optimizer2.step()

            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data).float()

        for i, (input1, input2, labels) in enumerate(tqdm(test_data_loader)):
            input1 = input1.cuda(CUDA)
            input2 = input2.cuda(CUDA)
            labels = labels.long().cuda(CUDA)

            with torch.no_grad():
                output1 = model(input1)
                output2 = model(input2)
                output = output_layer(output1, output2)

            preds = torch.argmax(output, dim=1)
            loss = loss_fn(output, labels)

            test_loss += loss.item()
            test_acc += torch.sum(preds == labels.data).float()

        if (test_loss/len(test_dataset)) < best :
            best = test_loss/len(test_dataset)
            # torch.save(model.state_dict(), os.path.join(SAVE_ROOT, f'{DATE}-feature.pth'))
            # torch.save(output_layer.state_dict(), os.path.join(SAVE_ROOT, f'{DATE}-output.pth'))

        train_loss = train_loss/len(train_dataset)
        train_loss_curve.append(train_loss)
        train_acc = train_acc/len(train_dataset)
        train_acc_curve.append(train_acc)

        test_loss = test_loss/len(test_dataset)
        test_loss_curve.append(test_loss)
        test_acc = test_acc/len(test_dataset)
        test_acc_curve.append(test_acc)

        print('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(train_loss, train_acc))
        print('Test loss: {:.4f}\taccuracy: {:.4f}\n'.format(test_loss, test_acc))
    visualize(train_loss_curve, train_acc_curve, test_loss_curve, test_acc_curve, 'new')



if __name__ == "__main__":
    train()
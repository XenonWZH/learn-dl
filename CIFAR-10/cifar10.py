import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    def __init__(self, data_path, labels_path=None):
        self.data_path = data_path
        
        self.data_name = []
        self.data_labels = []
        if labels_path:
            self.labels = True
            csv = pd.read_csv(labels_path)
            self.data_name = [f'{name}.png' for name in csv['id'].tolist()]
            self.data_labels = csv['label'].tolist()
            self.labels_set = list(set(self.data_labels))
            self.labels_map = {label: idx for idx, label in enumerate(self.labels_set)}
            self.data_labels = [self.labels_map[label] for label in self.data_labels]
        else:
            self.labels = False
            self.data_name = sorted([name for name in os.listdir(data_path) if name.endswith('.png')])

        self.len = len(self.data_name)

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.data_name[index])
        img = Image.open(path).convert('RGB')

        img = self.transform(img)

        return (img, self.data_labels[index]) if self.labels else img
    
    def get_labels_set(self):
        return self.labels_set

batch_size = 50

train_dataset = CIFAR10Dataset(data_path='../data/kaggle_cifar10_tiny/train', labels_path='../data/kaggle_cifar10_tiny/trainLabels.csv')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
test_dataset = CIFAR10Dataset(data_path='../data/kaggle_cifar10_tiny/test')
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)
    
class Net(nn.Module):
    def __init__(self, types):
        super(Net, self).__init__()
        self.types = types
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(800, types)  # 800 = 32 * 5 * 5
        self.relu = nn.ReLU()

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(self.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(self.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

model = Net(types=len(train_dataset.get_labels_set()))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 5 == 4:
            print('[%d, %4d] loss: %.6f' % (epoch + 1, (batch_idx + 1) * batch_size, running_loss / 5))
            running_loss = 0.0

def test():
    model.eval()
    epoch = 0

    with torch.no_grad():
        for data in test_loader:
            imgs = data.to(device)
            labels = model(imgs)
            for label in labels.argmax(dim=1).cpu().numpy():
                epoch += 1
                print(f'#{epoch}: {train_dataset.get_labels_set()[label]}')

if __name__ == '__main__':
    print('--- Training ---')
    for epoch in range(500):
        train(epoch)
    
    print()
    
    print('--- Testing ---')
    test()
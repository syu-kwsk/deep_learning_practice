import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, batch_size):
        self.batch_size = batch_size

        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256*2*2, 1)

    def forward(self, x):
        x = F.avg_pool2d(F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True), 2)
        x = F.avg_pool2d(F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True), 2)
        x = F.avg_pool2d(F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True), 2)
        x = F.avg_pool2d(F.leaky_relu(self.bn4(self.conv4(x)), 0.2, inplace=True), 2)
        x = self.fc(x.view(self.batch_size, -1))
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convt1 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.convt4 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.convt1(x)), inplace=True)
        x = F.relu(self.bn2(self.convt2(x)), inplace=True)
        x = F.relu(self.bn3(self.convt3(x)), inplace=True)
        x = torch.tanh(self.convt4(x))
        return x

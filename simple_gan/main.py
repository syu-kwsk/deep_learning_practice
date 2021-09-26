import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

from model import Generator, Discriminator
epochs = 500
batch_size = 2048

transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = torchvision.datasets.CIFAR10(root="./data",
                                            train=True, download=True,
                                            transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_g = Generator().to(device)
model_d = Discriminator(batch_size).to(device)

crit = nn.BCEWithLogitsLoss()

opt_g = optim.Adam(model_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = optim.Adam(model_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

const_z = torch.randn(64, 128, 1, 1).to(device)
loss_list = []
max_loss_g = 0.0
min_loss_d = 10.0

for e in range(1, epochs+1):
    model_g.train()
    model_d.train()
    running_loss_g = 0.0
    running_loss_d = 0.0
    for i, (real_img, _) in enumerate(dataloader):
        if len(real_img) != batch_size:
            continue

        ones = torch.ones(batch_size, dtype=torch.float).view(batch_size, 1).to(device)
        zeros = torch.zeros(batch_size, dtype=torch.float).view(batch_size, 1).to(device)

        # Generatorの学習
        z = torch.randn(batch_size, 128, 1, 1).to(device)
        fake_img = model_g(z)
        fake_output = model_d(fake_img)
        loss_g = crit(fake_output, ones)

        model_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # Discriminatorの学習
        real_img = real_img.to(device)
        real_output = model_d(real_img)
        real_loss_d = crit(real_output, ones)

        fake_output = model_d(fake_img.detach())
        fake_loss_d = crit(fake_output, zeros)
        loss_d = real_loss_d + fake_loss_d

        model_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        running_loss_g += loss_g.item()
        running_loss_d += loss_d.item()

        if i%25 == 0:
            print('[Epoch:{}/{}][{}/{}] Loss_g:{:.5f} Loss_d:{:.5f}'.format(e, epochs, i, int(len(dataloader)), loss_g.item(), loss_d.item()))

    loss_list.append([running_loss_g/int(len(dataloader)), running_loss_d/int(len(dataloader))])

    if max_loss_g < running_loss_g/int(len(dataloader)):
        max_loss_g = running_loss_g/int(len(dataloader))
        torch.save(model_g.state_dict(), 'model_g.pth')
        print("save model g")

    if min_loss_d > running_loss_d/int(len(dataloader)):
        min_loss_d = running_loss_d/int(len(dataloader))
        torch.save(model_d.state_dict(), 'model_d.pth')
        print("save model d")

    if e%5 == 0:
        model_g.eval()
        print('save img of epoch {}'.format(e))
        if not os.path.exists("save_images"):
            os.mkdir("save_images")
        output_img = model_g(const_z)
        save_image(output_img, "save_images/{}.jpg".format(e))

epoch_list = [i+1 for i in range(len(loss_list))]
plts = plt.plot(epoch_list, loss_list)
plt.legend(plts, ('generator loss', 'discriminator loss'))
plt.savefig('learning_curve.png')

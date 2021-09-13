import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from tqdm import tqdm

from model import cnn

def train(model, dataloader, device):
    # model = model.train()
    num_epochs = 1
    criterion=nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        with torch.set_grad_enabled(True):
            loss_sum=0
            corrects=0
            total=0

            with tqdm(total=len(dataloader), unit="batch") as pbar:
                    pbar.set_description(f"Epoch[{epoch}/{num_epochs}]")
                    for imgs,labels in dataloader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        output=model.forward(imgs)
                        loss=criterion(output,labels)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        predicted = torch.argmax(output, dim = 1)
                        corrects += (predicted == labels).sum()
                        total += imgs.size(0)

                        #loss関数で通してでてきたlossはCrossEntropyLossのreduction="mean"なので平均
                        #batch sizeをかけることで、batch全体での合計を今までのloss_sumに足し合わせる
                        loss_sum += loss*imgs.size(0)

                        accuracy = corrects.item() / total
                        running_loss = loss_sum/total
                        pbar.set_postfix({"loss": running_loss.item(), "accuracy": accuracy })
                        pbar.update(1)

def test(model, dataloader, device):
    total, tp = 0, 0
    for (x, label) in dataloader:

        # to CPU
        x = x.to(device)

        # 推定
        y_ = model.forward(x)
        label_ = y_.argmax(1).to('cpu')

        # 結果集計
        total += label.shape[0]
        tp += (label_==label).sum().item()

    acc = tp/total
    print('accuracy = %.3f' % acc)

if __name__ == '__main__':
    # GPU or CPUの自動判別
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device) # cpu

    # modelの定義
    model = cnn().to(device)
    opt = torch.optim.Adam(model.parameters())

    # datasetの読み出し
    bs = 128 # batch size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False)

    train(model.train(), trainloader, device)
    test(model.eval(), testloader, device)

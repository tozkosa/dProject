import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

"""paramters"""
train_dataroot = "../daon_data/ishikawa_with_fake/"
test_dataroot = "../daon_data/nagoya_data/"
batch_size = 64
num_classes = 2
num_epochs = 1000
lr = 0.01
check_interval = 100 

"""paramters end."""


train_dataset = ImageFolder(train_dataroot, transform=transforms.ToTensor())
test_dataset = ImageFolder(test_dataroot, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 3, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, num_classes)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MLPNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    #train
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.view(-1, 28*28*3).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        loss.backward()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)

    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.view(-1, 28*28*3).to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)

    print('Epoch [{}/{}], ({}), Loss:{loss:.4f}, train_acc: {train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}'
    .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, train_acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

    if (epoch + 1) % check_interval == 0:
        checkpoint = {'epoch': epoch+1,
                      'mlp_state_dict': net.state_dict(),
                      'opt_state_dict': optimizer.state_dict()
        }
        outfile = f'../daon_checkpoint/mlp_epoch_{epoch+1}.cpt'
        torch.save(checkpoint, outfile)

plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()

plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()

plt.show()





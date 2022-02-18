import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchsummary

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt

#from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import axes3d

import pandas as pd

"""paramters"""
train_dataroot = "../daon_data/ishikawa_data/"  #"../daon_data/ishikawa_with_fake/"
test_dataroot = "../daon_data/nagoya_data/" #"../daon_data/nagoya_data/"
batch_size = 64
num_classes = 2
num_epochs = 500 #200
lr = 0.01
check_interval = 10
"""paramters end."""


train_dataset = ImageFolder(train_dataroot, transform=transforms.ToTensor())
test_dataset = ImageFolder(test_dataroot, transform=transforms.ToTensor())

# for i in range(100,200):
#     image, label = train_dataset[i]
#     print(image.size())
#     print(label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

""" for images, labels in train_loader:
    print(images.size())
    print(images[0].size())
    print(labels.size())
    for i in range(32):
        print(labels[i])
    break
 """

class LeNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=2), #11, 4, 5
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(400, 120),
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.Dropout(0.2),
            nn.Linear(84, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def learning(device):
    net = LeNet().to(device)

    torchsummary.summary(net, (3, 28, 28))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) #0.01

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
            images, labels = images.to(device), labels.to(device)
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
                images, labels = images.to(device), labels.to(device)
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
            checkpoint = {'epoch': epoch + 1,
                          'lenet_state_dict': net.state_dict(),
                          'opt_state_dict': optimizer.state_dict()
                          }
            outfile = f'../daon_checkpoint/lenet_epoch_{epoch+1}.cpt'
            torch.save(checkpoint, outfile)

    loss_acc = {'train_loss': train_loss_list,
                'train_acc': train_acc_list,
                'val_loss': val_loss_list,
                'val_acc': val_acc_list}
    return loss_acc

def render_frame(angle, X_r, y):
    """data の 3D 散布図を PIL Image に変換して返す"""
    #global data
    colors = ['navy', 'magenta']
    target_names = ['defect', 'normal']
    lw = 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for color, i, target_name in zip (colors, [0, 1], target_names):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], color=color, alpha=0.8, lw=lw,
                    label=target_name)
    ax.view_init(30, angle)
    plt.close()
    # 軸の設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    """ ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3) """
    # PIL Image に変換
    buf = BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    return Image.open(buf)

def graph_learning(loss_acc):
    plt.figure()
    plt.plot(range(num_epochs), loss_acc['train_loss'], color='blue', linestyle='-', label='train_loss')
    plt.plot(range(num_epochs), loss_acc['val_loss'], color='green', linestyle='-', label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.grid()

    plt.figure()
    plt.plot(range(num_epochs), loss_acc['train_acc'], color='blue', linestyle='-', label='train_acc')
    plt.plot(range(num_epochs), loss_acc['val_acc'], color='green', linestyle='-', label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('Training and validation accuracy')
    plt.grid()

    plt.show()

def obtain_features(cptfile):
    cpt = torch.load(cptfile)
    stdict_net = cpt['lenet_state_dict']
    print(device)
    net_pretrained = LeNet().to(device)
    net_pretrained.load_state_dict(stdict_net)
    nagoya_data_list = np.empty((0,400))
    nagoya_label = np.empty(0)
    net_pretrained.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            x = net_pretrained.features(images)
            x = x.view(x.size(0), -1)
            #print(x.shape)
            nagoya_data_list = np.append(nagoya_data_list, x.to('cpu').detach().numpy(), axis=0)
            nagoya_label = np.append(nagoya_label, labels.to('cpu').detach().numpy())
    #print(nagoya_data_list)
    nagoya_data = np.array(nagoya_data_list)
    return nagoya_data, nagoya_label

def display_features(cptfile):
    cpt = torch.load(cptfile)
    stdict_net = cpt['lenet_state_dict']
    print(device)
    net_pretrained = LeNet().to(device)
    net_pretrained.load_state_dict(stdict_net)
    nagoya_data_list = np.empty((0,400))
    nagoya_label = np.empty(0)
    net_pretrained.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            x = net_pretrained.features(images)
            x = x.view(x.size(0), -1)
            print(x.shape)
            nagoya_data_list = np.append(nagoya_data_list, x.to('cpu').detach().numpy(), axis=0)
            nagoya_label = np.append(nagoya_label, labels.to('cpu').detach().numpy())
    print(nagoya_data_list)
    nagoya_data = np.array(nagoya_data_list)
    #print(nagoya_data.shape)
    #print(nagoya_label.shape)
    pca = PCA(n_components=3)
    X_r = pca.fit(nagoya_data).transform(nagoya_data)
    print(X_r)
    y = nagoya_label
    print(f'explained variance ratio (first two components):components):{pca.explained_variance_ratio_}')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = ['navy', 'magenta']
    target_names = ['defect', 'normal']
    lw = 0.5
    for color, i, target_name in zip (colors, [0, 1], target_names):
        ax.scatter(X_r[y == i, 0], X_r[y == i, 1], X_r[y == i, 2], color=color, alpha=0.2, lw=lw,
                    label=target_name)
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of Nagoya')
    plt.show()

    """ 
    render_frame(30, X_r, y)
    images = [render_frame(angle, X_r, y) for angle in range(360)]
    images[0].save('output.gif', save_all=True, append_images=images[1:], duration=100, loop=0) 
    """



if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    which  = 0 #1: learning 0:display

    if which:
        loss_acc = learning(device)
        graph_learning(loss_acc)

    else:
        ckpt = 200
        print(f'check point = {ckpt}')
        cptfile = f"../daon_checkpoint/lenet_epoch_{ckpt}.cpt"
        display_features(cptfile)
        """ features, labels = obtain_features(cptfile)
        print(features.shape)
        print(labels.shape)
        num_components = 2
        gmm = GaussianMixture(num_components, n_init=10).fit(features)
        prediction = gmm.predict(features)
        proba = gmm.predict_proba(features)
        print(labels)
        data = {'true': labels, 'probability': proba[:,0]}
        df = pd.DataFrame(data)
        df.to_csv('test.csv', encoding='utf-8') """





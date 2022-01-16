import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchsummary


# オリジナルデータを扱うためのクラス
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

num_gen = 1000  # fakeデータ数
cptfile = '../cGAN_results/checkpoint_epoch_10000.cpt'

cpt = torch.load(cptfile)
init_epoch = cpt['epoch']
stdict_netG = cpt['generator_state_dict']
stdict_netD = cpt['discriminator_state_dict']
stdict_optG = cpt['optG_state_dict']
stdict_optD = cpt['optD_state_dict']
G_losses = cpt['Gloss']
D_losses = cpt['Dloss']
D_x_out = cpt['Dx']
D_G_z1_out = cpt['DGz']
fixed_noise_label = cpt['f_noise_label']

# 設定

dataroot = "../daon_data/ishikawa_data/"

workers = 2

#####
batch_size=32  

# 潜在変数の次元
nz = 100

# Size of feature maps in generator
nch_g = 128

# Size of feature maps in discriminator
nch_d = 128

# エポック数
n_epoch = 8000    # 50

lr = 0.000001

beta1 = 0.5
outf = './cGAN_results'
display_interval = 600

# new for CGAN
n_class = 2

# check point interval
check_interval = 50

dataset = ImageFolder(dataroot, transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Grayscale(),
                                                              transforms.Normalize((0.5,), (0.5,)) ]))
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=int(workers))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    """
    Generator class
    """
    def __init__(self, nz=100, nch_g=128, nch=1):
        """
        :nz: 入力ベクトルzの次元
        :nch_g: 最終層の入力チャネル数
        :nch: 出力画像のチャネル数
        """
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 4, 3, 1, 0),    
                nn.BatchNorm2d(nch_g * 4),                     
                nn.ReLU()                                      
            ),  # (B, nz, 1, 1) -> (B, nch_g*4, 3, 3)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 3, 2, 0),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()
            ),  # (B, nch_g*4, 3, 3) -> (B, nch_g*2, 7, 7)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),  # (B, nch_g*2, 7, 7) -> (B, nch_g, 14, 14)
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            )   # (B, nch_g, 14, 14) -> (B, nch, 28, 28)
        })

    def forward(self, z):
        """
        :z: 入力ベクトル
        :return: 生成画像
        """
        for layer in self.layers.values():  # self.layersの各層で演算を行う
            z = layer(z)
        return z

# Generator
netG = Generator(nz=nz+n_class, nch_g=nch_g).to(device) # changed for cGAN
# initialize with weights_init function
# netG.apply(weights_init)  

# 学習済みパラメータの読み込み
netG.load_state_dict(stdict_netG)
#print(netG)

def onehot_encode(label, device, n_class=2):
    """create one-hot vector"""
    eye = torch.eye(n_class, device=device)

    # (B, c_class, 1, 1)?
    return eye[label].view(-1, n_class, 1, 1)

def concat_image_label(image, label, device, n_class=2):
    """concatenate image and label """
    B, C, H, W = image.shape
    oh_label = onehot_encode(label, device)
    oh_label = oh_label.expand(B, n_class, H, W)
    return torch.cat((image, oh_label), dim=1)

def concat_noise_label(noise, label, device):
    """concatenate noise and label"""
    oh_label = onehot_encode(label, device)
    return torch.cat((noise, oh_label), dim=1)

#noise = torch.randn(num_gen, nz, 1, 1, device=device)
noise = torch.randn(num_gen, nz, 1, 1, device=device)
# label for fake image, added for cGAN
fake_label = torch.randint(n_class, (num_gen,), dtype=torch.long, device=device)
# concatenate noise and label, added for cGAN
fake_noise_label = concat_noise_label(noise, fake_label, device)

fake_image = netG(fake_noise_label)
fake_image.shape
save_path = "../cGAN_results/test.png"
vutils.save_image(fake_image.detach(), save_path, normalize=True, nrow=num_gen)

plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake_image.to(device)[:64], 
                                         padding=2, 
                                         normalize=True).cpu(),(1,2,0)))

plt.show()

daon_class = ['normal', 'defect']

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v== val]

for i, (img, label) in enumerate(zip(fake_image, fake_label)):
    folder = get_keys_from_value(dataset.class_to_idx, label)
    savef = folder[0]
    save_path = f"../cGAN_results/fake_daon/{savef}/fake_{i}.png"
    print(save_path)
    vutils.save_image(img.detach(), save_path, normalize=True)



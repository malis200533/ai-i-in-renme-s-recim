import os 
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.gen = nn.Sequential(

            nn.Unflatten(1,(100,1,1)), #100x1x1

            nn.ConvTranspose2d(100,1024,4,1,0),
            nn.BatchNorm2d(1024),
            nn.ReLU(), # 1024x4x4

            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(), # 512x8x8

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # 256x16x16

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # 128x32x32

            nn.ConvTranspose2d(128,3,4,2,1),
            nn.Tanh() #3x64x64

        )

    def forward(self,x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(

            nn.Conv2d(3,64,4,2,1),
            nn.LeakyReLU(0.2), # 8x32x32 

            nn.Conv2d(64,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128), # 16x16x16
            
            nn.Conv2d(128,256,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256), # 32x8x8

            nn.Conv2d(256,512,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512), # 64x4x4

            nn.Conv2d(512,1,4,2,0),
            nn.Flatten(),
            nn.Sigmoid() # 1x1x1

        )
    
    def forward(self,x):
        return self.disc(x)
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.LeakyReLU(0.2), # 8x32x32 

            nn.Conv2d(64,128,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128), # 16x16x16
            
            nn.Conv2d(128,256,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256), # 32x8x8

            nn.Conv2d(256,512,4,2,1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512), # 64x4x4

            nn.Conv2d(512,100,4,2,0),
            nn.Flatten(),
            nn.Sigmoid() # 1x1x1
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1,(100,1,1)), #100x1x1

            nn.ConvTranspose2d(100,1024,4,1,0),
            nn.BatchNorm2d(1024),
            nn.ReLU(), # 1024x4x4

            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(), # 512x8x8

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # 256x16x16

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # 128x32x32

            nn.ConvTranspose2d(128,3,4,2,1),
            nn.Tanh() #3x64x64
        )
    
    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)



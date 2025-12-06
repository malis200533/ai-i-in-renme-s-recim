import torch
from torch import nn

class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim,hidden_dim= 200, z_dim=20):
        super(VariationalAutoEncoder,self).__init__()

        # burası encoder için yazılan yerler
        self.img2hid = nn.Linear(input_dim,hidden_dim)
        self.hid2mu = nn.Linear(hidden_dim,z_dim)
        self.hid2sigma = nn.Linear(hidden_dim,z_dim)

        # burası da decoder için yazılan yerler
        self.z2hid = nn.Linear(z_dim, hidden_dim)
        self.hid2img = nn.Linear(hidden_dim,input_dim)

        self.relu = nn.ReLU()
        self.sigmo = nn.Sigmoid()

    def encode(self,x):
        h = self.relu(self.img2hid(x))

        mu,sigma = self.hid2mu(h),self.hid2sigma(h)

        return mu,sigma

    def decode(self,z):
        h = self.relu(self.z2hid(z))
        return self.sigmo(self.hid2img(h))


    def forward(self,x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_parametrized = mu+sigma*epsilon
        x_reconstructed = self.decode(z_parametrized)

        return x_reconstructed, mu,sigma

if __name__ == "__main__":
    x = torch.randn(1,784)
    vae = VariationalAutoEncoder(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape)
    print(mu.shape)
    print(sigma.shape)

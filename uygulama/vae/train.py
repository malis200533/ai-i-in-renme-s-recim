import torch
from torch import nn
from torchvision import datasets,transforms
from encoder import VariationalAutoEncoder
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 784
h_dim = 200
z_dim = 20
num_epochs = 5
batch_size = 32
lr_rate = 3e-4

dataset = datasets.MNIST("variational_encoder/veri",True,transforms.ToTensor(),None,True)

data_loder = DataLoader(dataset,batch_size,True)

model = VariationalAutoEncoder(input_dim,h_dim,z_dim).to(device)

optim = torch.optim.Adam(model.parameters(),lr_rate)

criter = nn.BCELoss(reduction="sum")


# eğitime başlıyoruz
model.train()

for epoch in range(num_epochs):
    loop = tqdm(enumerate(data_loder))

    for i, (imgs,_) in loop:

        imgs = imgs.to(device).view(-1,784)
        
        reconstructed, mu, sigma = model(imgs)

        reconst_loss = criter(reconstructed, imgs)

        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        total_loss = reconst_loss+kl_div

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        loop.set_postfix(loss=total_loss.item())

images = []
idx = 0 

for x,y in dataset:
    if y == idx:
        images.append(x)
        idx += 1
    if y == 10:
        break

encodings_digit = []

model.to(torch.device("cpu"))
model.eval()

for d in range(10):
    mu, sigma = model.encode(images[d].view(1,784))
    encodings_digit.append((mu,sigma))

for sayi, (mu,sigma) in enumerate(encodings_digit):
    fakes = []
    for example in range(9):
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon*sigma

        out = model.decode(z)
        out = out.view(1,1,28,28)
        fakes.append(out)

    fakes = torch.cat(fakes,0)
    fakes = make_grid(fakes,3,normalize=False)
    save_image(fakes,f"variational_encoder/cikti/generated_{sayi}.jpg")
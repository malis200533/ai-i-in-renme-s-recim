import os 
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image

from dcganlib import Generator, Discriminator,Autoencoder, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = datasets.ImageFolder("veriler/resimler", trans)

loder = DataLoader(dataset,64,True)

gen = Generator().to(device)
disc = Discriminator().to(device)
encoder = Autoencoder().to(device)
encoder.load_state_dict(torch.load("dcgan/autoencoder/encoder.pth"))

initialize_weights(gen)
initialize_weights(disc)


opt_gen = torch.optim.Adam(gen.parameters(), 2e-4, (0.5,0.999))
opt_disc = torch.optim.Adam(disc.parameters(), 2e-4 ,(0.5,0.999))

criter = nn.BCELoss()

step = 0

gen.train()
disc.train()
encoder.eval()

sabit_noise,_ = next(iter(loder))
sabit_noise = sabit_noise[:16]
sabit_noise = torch.tensor(sabit_noise).to(device)

sabit_noise = encoder.encoder(sabit_noise)

# sabit_noise = torch.randn_like(sabit_noise)*0.3 + sabit_noise
# sabit_noise = torch.clip(sabit_noise,0.0,1.0)

epochs = 5000

for epoch in range(epochs):

    g_loss = 0
    d_loss = 0

    bacther = 0

    for idx, (real, _) in enumerate(loder):
        real = real.to(device)

        batch_size = real.size(0)

        noise = encoder.encoder(real)

        disc_real = disc(real)
        loss_d_real = criter(disc_real, torch.ones_like(disc_real)*0.90)

        fakes = gen(noise)
        
        disc_fake = disc(fakes)
        loss_d_fake = criter(disc_fake, torch.zeros_like(disc_fake))

        loss_d = (loss_d_fake + loss_d_real)/2

        opt_disc.zero_grad()
        loss_d.backward(retain_graph=True)
        opt_disc.step()

        output = disc(fakes)
        loss_g = criter(output,torch.ones_like(output))

        opt_gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        g_loss += loss_g.item()
        d_loss += loss_d.item()

        bacther += 1
    
    g_loss = g_loss/bacther
    d_loss = d_loss/bacther

    print(f"Epoch: {epoch}/{epochs}, Loss D: {d_loss}, Loss G: {g_loss}")

    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            test_imgs = gen(sabit_noise)
            img_grid = make_grid(test_imgs,4,normalize=True)

            save_image(img_grid,f"veriler/cikti/dcgan{epoch}.jpg")
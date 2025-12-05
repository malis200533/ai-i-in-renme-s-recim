import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image

from dcganlib import Autoencoder, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])


dataset = datasets.ImageFolder("veriler/resimler",trans)

data_loder = DataLoader(dataset,32,True)

model = Autoencoder().to(device)

criter = nn.L1Loss()

opti_coder = torch.optim.Adam(model.parameters(),2e-4)


model.train()
for i in range(200):
    total = 0
    step = 0
    for imgs,_ in data_loder:
        imgs = imgs.to(device)

        out = model(imgs)

        loss = criter(out,imgs)

        total += loss.item()
        step += 1

        opti_coder.zero_grad()
        loss.backward()
        opti_coder.step()
    
    print(i,total/step)
        

model.eval()

with torch.no_grad():
    for img,_ in data_loder:
        img = img.to(device)
        fake = model(img)
        img = make_grid(img,normalize=True)
        fake = make_grid(fake,normalize=True)
        save_image(img,"veriler/cikti/real.jpg")
        save_image(fake,"veriler/cikti/fake.jpg")
        break

torch.save(model.state_dict(),"dcgan/autoencoder/encoder.pth")

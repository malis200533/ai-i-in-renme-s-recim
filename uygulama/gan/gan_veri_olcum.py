from torchvision import datasets,transforms
import torch
from torch.utils.data import DataLoader 

transform_sb = transforms.Compose(
    [transforms.Grayscale(),transforms.Resize((64,64)),transforms.ToTensor()]
)

dataset = datasets.ImageFolder("google_foto//corolla",transform_sb)

loader = DataLoader(dataset,32,False)

toplam_deger = 0.0
toplam_kare = 0.0
toplam_sayi = 0

for images,_ in loader:
    print(_)
    flatten = images.view(-1)
    piksel_say = flatten.size(0)
    toplam_deger += torch.sum(flatten)
    toplam_kare += torch.sum(flatten**2)

    toplam_sayi += piksel_say


mean_value = toplam_deger/toplam_sayi
mean_squar = toplam_kare/toplam_sayi

varyans = mean_squar - (mean_value**2)

std_value = torch.sqrt(varyans)

print(f"toplam piksel: {toplam_sayi}")
print(f"ortalama hesaplandı: {mean_value.item()}")
print(f"standart sapma: {std_value.item()}")

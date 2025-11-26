import os
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import classification_report , confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


#burada cihaz seçimi yapılmaktadır daha hızlı ve verimli eğitim için cudayı seçeceğiz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"kullanılan cihaz = {device}")

#burada veri yüklemesi için transform tanımlamaktayız 64x64 fotolar haline getirilip normalize edilecektir
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])


#aşağıda veri yüklenmesi ve ayarlanması gerçekleştirilir
dataset = datasets.ImageFolder(root="flower_photos",transform=transform)



train_size = int(0.7*len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_data,32,True,num_workers=0)
test_loader = DataLoader(test_data,32,False,num_workers=0)


#şimdi sınıflandırma yapması için eğitilecek modelimizi oluşturalım

class simpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(simpleCNN,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(32*16*16,128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.model(x)
    
#program için şimdi modelimizi çağıralım

model = simpleCNN(5).to(device)

#kayıp fonksiyonu ve optimizasyon yöntemini yazalım

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epochs = 10

#şimdi modelimizi eğitmeye başlayabiliriz

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    print(f"epoch {epoch+1}/{epochs}, loss: {running_loss/len(train_loader)}")

#yukarıda modelimizi eğitirken cross entropy loss bizim için 0dan 4e kadar labelları 5 değerli vektör çıktımıza otomatik olarak uyarlayacaktır 
#bunu yaparken indexlerin hangi sınıfa ait olduğunu dataset classına dataset.claas_to_idx yazarak sonucu alabiliri
#veya datasetle ilgili daha başka verileri de benzer şekilde alabiliriz

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _,preds = torch.max(outputs,1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels,all_preds,target_names=dataset.classes))


cm = confusion_matrix(all_labels,all_preds)
plt.figure(figsize=(6,5))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=dataset.classes,yticklabels=dataset.classes)

plt.xlabel("tahmin")
plt.ylabel("gerçek")

plt.title("confusion matrix")

plt.tight_layout()
plt.show()

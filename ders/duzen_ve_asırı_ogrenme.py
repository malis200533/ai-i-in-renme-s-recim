import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split

from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

dataset = datasets.ImageFolder("flower_photos",trans)

train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size

train_data,val_data = random_split(dataset,[train_size,val_size])

train_loder = DataLoader(train_data,32,True)
val_loder = DataLoader(val_data,32,False)

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),

            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*32*32,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,num_classes),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv(x)
        return self.classifier(x)
    

model = CNN(len(dataset.classes)).to(device)

criter = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)


best_val_acc = 0.0
patience = 3
trigger_times = 0

for epoch in range(20):
    model.train()
    for images,labels in train_loder:
        images,labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = model(images)
        loss = criter(outputs,labels)
        loss.backward()
        optimizer.step()
    

    model.eval()
    preds,targets = [],[]
    with torch.no_grad():
        for images,labels in val_loder:
            images = images.to(device)
            outputs = model(images)
            _,predict = torch.max(outputs,1)
            preds.extend(predict.cpu().numpy())
            targets.extend(labels.numpy())

    val_acc = accuracy_score(targets,preds)
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc}")

    

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
    
    else:
        trigger_times+=1
        if trigger_times > patience:
            print("early stopped")
            break

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n Toplam Öğrenilebilir Parametre: {total_params}")
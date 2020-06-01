import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from model.position_net import PositionNet


EPOCH = 1000
lr = 0.001

train_transforms = transforms.Compose([transforms.RandomRotation(10),
 transforms.ToTensor()])

image_data = ImageFolder('data/data',transform=train_transforms)
data_loader = DataLoader(image_data, batch_size=10, shuffle=True)
class_names = image_data.classes

model = PositionNet(output=6)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
losses = []

for i in range(EPOCH):

    for idx, (images, labels) in enumerate(data_loader):

        y_pred = model.forward(images)

        loss = criterion(y_pred, labels)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%10==0:
        print(f"Trained epoch: {i} with a loss of: {loss}")

torch.save(model.state_dict(), "PositionNet.pt")

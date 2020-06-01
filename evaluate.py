import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model.position_net import PositionNet

model = PositionNet()
model.load_state_dict(torch.load("PositionNet.pt"))

test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

print("Loading testing data. ")

test_data = ImageFolder('data/data', transform=test_transforms)
test_data_loader = DataLoader(test_data)

print("Comparing prediction to testing labels")

with torch.no_grad():
    correct = 0
    total = 0
    for i, (data, labels) in enumerate(test_data_loader):

        y_pred = model.forward(data)
        if y_pred.argmax().item() == labels:
            correct += 1
        total += 1

print(f"The accuracy of the loaded model is {correct*100 / total}%. ")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchvision.models import resnet18
from model import TinyResNet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = TinyResNet().to(device)
print(sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


def compute_fim(loader, model, criterion):
    fim = None
    model.eval()
    for images, labels in tqdm(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        grad_flat = torch.cat(gradients)
        outer_prod = torch.ger(grad_flat, grad_flat)

        if fim is None:
            fim = outer_prod
        else:
            fim += outer_prod

    fim /= len(loader.dataset)
    return fim

fim = compute_fim(test_loader, model, criterion)
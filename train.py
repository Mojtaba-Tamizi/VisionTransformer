import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from model import VisionTransformer
from tqdm.auto import tqdm

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("NumPy version:", np.__version__)
print("CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

IMAGE_SIZE = 32
IN_CHANNELS = 3
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 3e-4
PATCH_SIZE = 4
NUM_CLASSES = 10
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
MLP_DIM = 512
DROPOUT_RATE = 0.1


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

print("Number of training samples:", len(train_dataset))
print("Number of test samples:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Number of training batches:", len(train_loader))
print("Number of test batches:", len(test_loader))


model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, IN_CHANNELS, NUM_CLASSES, 
                          EMBED_DIM, NUM_HEADS, NUM_LAYERS, MLP_DIM, DROPOUT_RATE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

def evaluate(model, device, test_loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)

            total_loss += loss.item() * x.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Epochs"):
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f'Epoch {epoch}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
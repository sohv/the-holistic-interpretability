import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from cnn_model import SmallCNN


def train(model, train_loader, epochs=5, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")


def test(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            _, pred = torch.max(out, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = SmallCNN()
    print('Training SmallCNN on Fashion-MNIST...')
    train(model, train_loader, epochs=5, device=device)

    acc = test(model, test_loader, device=device)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/smallcnn_fashionmnist.pth')
    print('Model saved to models/smallcnn_fashionmnist.pth')

if __name__ == '__main__':
    main()
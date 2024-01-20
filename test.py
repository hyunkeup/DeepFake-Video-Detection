import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.DfdcDataset import DfdsDataset

INPUT_FRAME_SHAPE: tuple = tuple([int(x) for x in os.environ.get("INPUT_FRAME_SHAPE").split(",")])


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(INPUT_FRAME_SHAPE[2], INPUT_FRAME_SHAPE[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(244 * 122 * 122, 2)  # 예시로 출력 뉴런 수를 2로 지정

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


transform = transforms.Compose([
    transforms.Resize((INPUT_FRAME_SHAPE[0], INPUT_FRAME_SHAPE[1])),
    transforms.ToTensor(),
])


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    print("=" * 40 + " Start training " + "=" * 40)
    print("Load dataset: ", end="")
    dataset = DfdsDataset(transform=transform)
    print("Done")
    print(f"* Classes: {dataset.classes}, Number of dataset: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(dataloader, model, criterion, optimizer)
        test(dataloader, model, criterion)
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    print("Done!")

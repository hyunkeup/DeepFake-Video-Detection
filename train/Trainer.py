import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, train_dataset: Dataset,
                 test_dataset: Dataset, batch_size=32,
                 num_of_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.batch_size = batch_size
        self.num_of_epochs = num_of_epochs

    def start(self):
        for epoch in range(self.num_of_epochs):
            print(f"Epoch {epoch + 1}\n--------------------------------------------------------------")
            self._train(epoch)
            self._test()

    def _train(self, epoch):
        # Train
        self.model.train()
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Prediction error calculation
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch == len(self.train_dataloader) - 1:
                print(
                    f'Training - Epoch [{epoch + 1}/{self.num_of_epochs}], Loss: {loss.item()}')

    def _test(self):
        # Test
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= self.batch_size
        correct /= len(self.test_dataloader.dataset)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def save_model(self, path: str = None, prefix=None):
        if path is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if prefix is None:
                path = f"model_{current_time}.pth"
            else:
                path = f"{prefix}_model_{current_time}.pth"

        torch.save(self.model.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")

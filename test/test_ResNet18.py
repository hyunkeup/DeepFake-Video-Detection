import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from dataset.DfdcDataset import ImageDataset
from property import Property

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
PREPROCESSED_DIRECTORY: str = Property.get_property("preprocessed_directory")
TEST_DIRECTORY: str = Property.get_property("test_directory")

if __name__ == "__main__":
    print("=" * 40 + " Start training " + "=" * 40)
    print("Load dataset: ", end="")
    test_dataset = ImageDataset(directory=TEST_DIRECTORY)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    print("Done")
    print(f"\t* Classes: {test_dataset.classes}, # of test: {len(test_dataset)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = models.resnet18().to(device)
    model.load_state_dict(torch.load("./ResNet18_model_20240120_164851.pth"))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Test
    correct = 0
    num_of_games = len(test_dataset)
    win_cnt = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= len(test_dataloader.dataset)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%")

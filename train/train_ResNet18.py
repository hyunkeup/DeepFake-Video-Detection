import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from dataset.DfdcDataset import ImageDataset
from property import Property
from train.Trainer import Trainer

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
PREPROCESSED_DIRECTORY: str = Property.get_property("preprocessed_directory")
TEST_DIRECTORY: str = Property.get_property("test_directory")

if __name__ == "__main__":
    print("=" * 40 + " Start training " + "=" * 40)
    batch_size = 32
    epochs = 100

    print("Load dataset: ", end="")
    train_dataset = ImageDataset(directory=PREPROCESSED_DIRECTORY)
    test_dataset = ImageDataset(directory=TEST_DIRECTORY)
    print("Done")
    print(f"\t* Classes: {train_dataset.classes}, # of train: {len(train_dataset)} / # of test: {len(train_dataset)}")

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = models.resnet18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer,
                      train_dataset=train_dataset, test_dataset=test_dataset,
                      batch_size=batch_size, num_of_epochs=epochs)
    trainer.start()
    trainer.save_model(prefix="ResNet18")
    print("Done!")

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from property import Property

PREPROCESSED_DIRECTORY: str = Property.get_property("preprocessed_directory")


class DfdsDataset(Dataset):
    def __init__(self, transform: transforms.Compose = None):
        """
        :param transform:
        """
        self.dataset = ImageFolder(root=f"{PREPROCESSED_DIRECTORY}", transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

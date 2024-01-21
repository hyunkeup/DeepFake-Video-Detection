import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from property import Property

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))


class ImageDataset(Dataset):
    def __init__(self, directory: str, transform: transforms.Compose = None):
        """
        :param transform:
        """
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((FRAME_SHAPE[0], FRAME_SHAPE[1])),
                transforms.ToTensor(),
            ])
        self.dataset = ImageFolder(root=f"{directory}", transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

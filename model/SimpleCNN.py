import torch.nn as nn

from property import Property

FRAME_SHAPE: tuple = tuple(Property.get_property("frame_shape"))
PREPROCESSED_DIRECTORY: str = Property.get_property("preprocessed_directory")
TEST_DIRECTORY: str = Property.get_property("test_directory")


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(FRAME_SHAPE[2], FRAME_SHAPE[0], kernel_size=3, stride=1, padding=1)
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

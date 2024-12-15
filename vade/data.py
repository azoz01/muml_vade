import pandas as pd
from torch import Tensor, flatten
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor

from .device import DEVICE

MNIST_TRANSFORMS = Compose(
    [ToTensor(), Normalize([0.13], [0.3]), Lambda(flatten), Lambda(lambda x: x.to(DEVICE))]
)

DATASETS = {
    "HAR": {
        "train": TensorDataset(
            Tensor(pd.read_csv("data/har/X_train.csv", sep=r"\s+", header=None).values).to(DEVICE),
            Tensor(pd.read_csv("data/har/y_train.csv", sep=r"\s+", header=None).values).to(DEVICE),
        ),
        "test": TensorDataset(
            Tensor(pd.read_csv("data/har/X_test.csv", sep=r"\s+", header=None).values).to(DEVICE),
            Tensor(pd.read_csv("data/har/y_test.csv", sep=r"\s+", header=None).values).to(DEVICE),
        ),
    },
    "MNIST": {
        "train": datasets.MNIST(
            root="data/mnist", download=True, train=True, transform=MNIST_TRANSFORMS
        ),
        "test": datasets.MNIST(
            root="data/mnist", download=True, train=False, transform=MNIST_TRANSFORMS
        ),
    },
}

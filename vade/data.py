import pandas as pd
from torch import Tensor, clamp, flatten
from torch.utils.data import TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor

MNIST_TRANSFORMS = Compose(
    [
        ToTensor(),
        Lambda(flatten),
        Lambda(lambda t: clamp(t, 1e-2, 1 - 1e-2)),
        Lambda(lambda x: x.to("cpu")),
    ]
)

DATASETS = {
    "HAR": {
        "train": TensorDataset(
            Tensor(
                pd.read_csv(
                    "data/har/X_train.csv", sep=r"\s+", header=None
                ).values
            ).to("cpu"),
            Tensor(
                pd.read_csv(
                    "data/har/y_train.csv", sep=r"\s+", header=None
                ).values
            ).to("cpu"),
        ),
        "test": TensorDataset(
            Tensor(
                pd.read_csv(
                    "data/har/X_test.csv", sep=r"\s+", header=None
                ).values
            ).to("cpu"),
            Tensor(
                pd.read_csv(
                    "data/har/y_test.csv", sep=r"\s+", header=None
                ).values
            ).to("cpu"),
        ),
    },
    "MNIST": {
        "train": datasets.MNIST(
            root="data/mnist",
            download=True,
            train=True,
            transform=MNIST_TRANSFORMS,
        ),
        "test": datasets.MNIST(
            root="data/mnist",
            download=True,
            train=False,
            transform=MNIST_TRANSFORMS,
        ),
    },
}

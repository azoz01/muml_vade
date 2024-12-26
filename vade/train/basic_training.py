from pathlib import Path

from torch.utils.data import DataLoader

from ..model import Model
from .training import Training


class BasicTraining(Training):

    def __init__(self, max_epochs: int, logs_path: Path, test: bool = False):
        self.trainer = self.get_trainer(max_epochs, logs_path, test)

    def train_model(
        self,
        model: Model,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> Model:
        return self.trainer.fit(model, train_dataloader, test_dataloader)

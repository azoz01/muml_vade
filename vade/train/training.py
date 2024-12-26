from abc import ABC, abstractmethod
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from ..model import Model


class Training(ABC):

    @abstractmethod
    def train_model(
        self,
        model: Model,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> Model:
        pass

    def get_trainer(
        self, max_epochs: int, logs_path: Path, test: bool = False
    ) -> pl.Trainer:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=3,
            verbose=False,
            mode="min",
        )
        model_checkpoint = ModelCheckpoint(
            filename="best_model-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_weights_only=True,
            every_n_epochs=1,
        )
        return pl.Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=1,
            callbacks=[early_stopping, model_checkpoint],
            fast_dev_run=test,
            log_every_n_steps=10,
            default_root_dir=logs_path,
        )

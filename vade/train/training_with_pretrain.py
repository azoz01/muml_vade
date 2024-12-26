from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader

from ..device import DEVICE
from ..model import AE, VADE, VAE, Model
from .training import Training


class TrainingWithPretrain(Training):

    def __init__(self, max_epochs: int, logs_path: Path, test: bool = False):
        self.pretrain_trainer = self.get_trainer(
            max_epochs, logs_path / "pretrain", test
        )
        self.trainer = self.get_trainer(max_epochs, logs_path / "train", test)

    def train_model(
        self,
        model: Model,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> Model:
        model_cls = type(model)
        assert model_cls in (
            VAE,
            VADE,
        ), "Pretraining strategy only supports VAE and VADE"
        ae = AE.from_model(model)
        self.pretrain_trainer.fit(ae, train_dataloader, test_dataloader)
        if Path(
            self.pretrain_trainer.checkpoint_callback.best_model_path
        ).is_file():
            logger.debug(
                f"Loading best checkpoint from {self.pretrain_trainer.checkpoint_callback.best_model_path}"  # noqa: E501
            )
            ae = AE.load_from_checkpoint(
                self.pretrain_trainer.checkpoint_callback.best_model_path
            )
        if model_cls is VADE:
            model = model_cls.from_ae(ae, model.n_classes).to(DEVICE)
        else:
            model = model_cls.from_ae(ae).to(DEVICE)
        self.trainer.fit(model, train_dataloader, test_dataloader)
        if Path(self.trainer.checkpoint_callback.best_model_path).is_file():
            logger.debug(
                f"Loading best checkpoint from {self.trainer.checkpoint_callback.best_model_path}"
            )
            model = model_cls.load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path
            )
        return model

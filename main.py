import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from vade.args import parse_args
from vade.config import Config
from vade.data import DATASETS
from vade.model_factory import get_model
from vade.train import BasicTraining

warnings.simplefilter("ignore")
torch.set_float32_matmul_precision("medium")

RESULTS_PATH = Path("results")


def main() -> None:
    pl.seed_everything(123)
    logger.info("Staring model training")
    args = parse_args()
    logger.debug(f"Parsed shell arguments: {args.__dict__}")

    logger.info("Parsing config")
    with open(args.config_path, "r") as f:
        config = Config.model_validate(yaml.load(f, Loader=yaml.CLoader))
    logger.debug(f"Parsed config: {config}")

    logger.info("Preparing data")
    train_dataset, test_dataset = (
        DATASETS[args.data]["train"],
        DATASETS[args.data]["test"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Preparing model")
    model = get_model(args.model, config, train_dataloader)
    model.compile()
    logger.debug(f"Model: {model}")

    logger.info("Training the model")
    training = BasicTraining(
        config.n_epochs, RESULTS_PATH / args.data / args.model, test=args.test
    )
    training.train_model(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()

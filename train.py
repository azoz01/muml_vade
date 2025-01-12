import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch._dynamo
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from vade.args import parse_args
from vade.config import Config
from vade.data import DATASETS
from vade.evaluate.vade import evaluate_vade
from vade.evaluate.vae import evaluate_vae
from vade.model import VADE
from vade.model_factory import get_model

torch._dynamo.config.suppress_errors = True
warnings.simplefilter("ignore")
torch.set_float32_matmul_precision("medium")

RESULTS_PATH = Path("results")
RESULTS_PATH = (
    RESULTS_PATH / os.environ.get("EXPERIMENT_SUBPATH", "")
    if "EXPERIMENT_SUBPATH" in os.environ
    else RESULTS_PATH
)


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
    output_path = (
        RESULTS_PATH / args.data / args.model / config.training.__name__
    )
    training = config.training(
        config.n_epochs,
        RESULTS_PATH / args.data / args.model / config.training.__name__,
        test=args.test,
    )
    training.train_model(model, train_dataloader, test_dataloader)

    logger.info("Evaluating model")
    if isinstance(model, VADE):
        evaluate_vade(model, test_dataloader, output_path, args.data)
    else:
        evaluate_vae(model, test_dataloader, output_path, args.data)


if __name__ == "__main__":
    main()

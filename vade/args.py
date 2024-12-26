from argparse import ArgumentParser, Namespace
from pathlib import Path

from .data import DATASETS
from .model import MODELS


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=list(sorted(MODELS.keys())),
        help="Name of the model to be trained",
        required=True,
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        choices=list(sorted(DATASETS.keys())),
        help="Name of the model to be trained",
        required=True,
    )
    parser.add_argument(
        "--config-path",
        "-c",
        type=Path,
        help="Name of the model to be trained",
        required=True,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use this to run in test mode",
    )
    return parser.parse_args()

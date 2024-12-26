from copy import deepcopy

import torch
from loguru import logger
from torch.utils.data import DataLoader

from .config import Config
from .device import DEVICE
from .model import MODELS, Model, ModelLiteral


def get_model(
    model_type_name: ModelLiteral, config: Config, train_dataloader: DataLoader
) -> Model:
    train_dataloader = deepcopy(train_dataloader)
    model_type = MODELS[model_type_name]
    input_data_dim = next(iter(train_dataloader))[0].shape[1]
    model_constructor_args = dict(
        layers_sizes=[input_data_dim] + config.hidden_layers_sizes,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        activation_function_cls=config.activation_function,
    )
    if model_type_name == "VADE":
        n_classes = torch.unique(
            torch.concat([b[1] for b in train_dataloader])
        )
        model_constructor_args["n_classes"] = n_classes.shape[0]
    logger.debug(f"Model constructor args: {model_constructor_args}")
    return model_type(**model_constructor_args).to(DEVICE).double()

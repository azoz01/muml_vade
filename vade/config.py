from typing import Type

from pydantic import BaseModel, field_validator
from torch import nn

from .train import TRAININGS, Training


class Config(BaseModel):
    hidden_layers_sizes: list[int]
    learning_rate: float
    weight_decay: float
    activation_function: str
    batch_size: int
    n_epochs: int
    training: str

    @field_validator("activation_function")
    def parse_activation_function(cls, v: str) -> Type[nn.Module]:
        if v not in dir(nn):
            raise ValueError(f"{v} should be a valid name of the nn.Module")
        return getattr(nn, v)

    @field_validator("training")
    def parse_training(cls, v: str) -> Type[Training]:
        if v not in TRAININGS.keys():
            raise ValueError(f"{v} should be in {list(TRAININGS.keys())}")
        return TRAININGS[v]

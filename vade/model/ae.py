from __future__ import annotations

from copy import deepcopy
from typing import Type

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class AE(pl.LightningModule):

    def __init__(
        self,
        layers_sizes: list[int],
        learning_rate: float,
        weight_decay: float,
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        assert (
            len(layers_sizes) >= 2
        ), "layers_sizes should have at least two elements"

        self.layers_sizes = layers_sizes
        self.activation_function_cls = activation_function_cls
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initialize_encoder(layers_sizes, activation_function_cls)
        self.initialize_decoder(layers_sizes, activation_function_cls)
        self.save_hyperparameters()

    def initialize_encoder(
        self,
        layers_sizes: list[int],
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ) -> None:
        encoder_modules = []
        for in_size, out_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            encoder_modules.append(nn.Linear(in_size, out_size))
            encoder_modules.append(activation_function_cls())
        self.encoder = nn.Sequential(*encoder_modules)

    def initialize_decoder(
        self,
        layers_sizes: list[int],
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ) -> None:
        reversed_layers_sizes = list(reversed(layers_sizes))
        decoder_modules = []
        for in_size, out_size in zip(
            reversed_layers_sizes[:-1], reversed_layers_sizes[1:]
        ):
            decoder_modules.append(nn.Linear(in_size, out_size))
            decoder_modules.append(activation_function_cls())
        decoder_modules.pop(-1)
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        X = X.double()
        return self.encoder(X)

    def decode(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(X)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        output = self.encode(X)
        output = self.decode(output)
        return output

    def training_step(self, batch: list[torch.Tensor, torch.Tensor], _):
        loss = self.loss_function(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor, torch.Tensor], _):
        loss = self.loss_function(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def loss_function(
        self, batch: list[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        X, _ = batch
        X_hat = self.forward(X).float()
        loss = F.mse_loss(X_hat, X, reduction="mean")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def from_model(model) -> AE:
        model = deepcopy(model)
        ae = AE(
            model.layers_sizes,
            model.learning_rate,
            model.weight_decay,
            model.activation_function_cls,
        )
        ae.encoder = model.encoder
        ae.decoder = model.decoder
        return ae.double()

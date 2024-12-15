from typing import Type

import pytorch_lightning as pl
import torch
from torch import nn


class VAE(pl.LightningModule):

    def __init__(
        self,
        layers_sizes: list[int],
        latent_dim: int,
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        assert len(layers_sizes) >= 2, "layers_sizes should have at least two elements"

        encoder_modules = []
        for in_size, out_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            encoder_modules.append(nn.Linear(in_size, out_size))
            encoder_modules.append(activation_function_cls())
        self.encoder = nn.Sequential(*encoder_modules)

        self.mu_encoder = nn.Linear(layers_sizes[-1], latent_dim)
        self.logvar_encoder = nn.Linear(layers_sizes[-1], latent_dim)

        reversed_layers_sizes = list(reversed(layers_sizes + [latent_dim]))
        decoder_modules = []
        for in_size, out_size in zip(reversed_layers_sizes[:-1], reversed_layers_sizes[1:]):
            decoder_modules.append(nn.Linear(in_size, out_size))
            decoder_modules.append(activation_function_cls())
        decoder_modules.pop(-1)
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.encoder(X)
        output = self._reparametrize(self.mu_encoder(output), self.logvar_encoder(output))
        output = self.decoder(output)
        return output

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

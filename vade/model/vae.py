from typing import Literal, Type

import pytorch_lightning as pl
import torch
from torch import nn


class VAE(pl.LightningModule):

    def __init__(
        self,
        layers_sizes: list[int],
        mode: Literal["AE", "VAE"],
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        assert len(layers_sizes) >= 2, "layers_sizes should have at least two elements"
        assert mode in ["AE", "VAE"], "mode should be either AE or VAE"

        self.mode = mode
        self.forward = self._forward_ae if mode == "AE" else self._forward_vae

        encoder_modules = []
        for in_size, out_size in zip(layers_sizes[:-1], layers_sizes[1:]):
            encoder_modules.append(nn.Linear(in_size, out_size))
            encoder_modules.append(activation_function_cls())
        self.encoder = nn.Sequential(*encoder_modules)

        self.mu_encoder = nn.Linear(layers_sizes[-1], layers_sizes[-1])
        self.logvar_encoder = nn.Linear(layers_sizes[-1], layers_sizes[-1])

        reversed_layers_sizes = list(reversed(layers_sizes))
        decoder_modules = []
        for in_size, out_size in zip(reversed_layers_sizes[:-1], reversed_layers_sizes[1:]):
            decoder_modules.append(nn.Linear(in_size, out_size))
            decoder_modules.append(activation_function_cls())
        decoder_modules.pop(-1)
        self.decoder = nn.Sequential(*decoder_modules)

    def _forward_ae(self, X: torch.Tensor) -> torch.Tensor:
        output = self.encoder(X)
        output = self.decoder(output)
        return output

    def to_ae(self):
        self.mode = "AE"
        self.forward = self._forward_ae

    def _forward_vae(self, X: torch.Tensor) -> torch.Tensor:
        output = self.encoder(X)
        output = self._reparametrize(self.mu_encoder(output), self.logvar_encoder(output))
        output = self.decoder(output)
        return output

    def to_vae(self):
        self.mode = "VAE"
        self.forward = self._forward_vae

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

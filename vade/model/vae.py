from __future__ import annotations

from copy import deepcopy
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from .ae import AE


class VAE(AE):

    def __init__(
        self,
        layers_sizes: list[int],
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(layers_sizes, activation_function_cls)
        self.mu_encoder = nn.Linear(layers_sizes[-1], layers_sizes[-1])
        self.logvar_encoder = nn.Linear(layers_sizes[-1], layers_sizes[-1])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.encode(X)
        output = self.reparametrize(self.mu_encoder(output), self.logvar_encoder(output))
        output = self.decode(output)
        return output

    def loss_function(self, batch: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        X, _ = batch
        encoded = self.encode(X)
        mu = self.mu_encoder(encoded)
        logvar = self.logvar_encoder(encoded)
        X_hat = self.decode(self.reparametrize(mu, logvar))
        recon_loss = F.mse_loss(X_hat, X, reduction="mean")
        kl_div_reg = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X.size(0)
        return recon_loss + kl_div_reg

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    @staticmethod
    def from_ae(ae: AE) -> VAE:
        vae = VAE(ae.layers_sizes, ae.activation_function_cls)
        vae.encoder = deepcopy(ae.encoder)
        vae.decoder = deepcopy(ae.decoder)
        return vae

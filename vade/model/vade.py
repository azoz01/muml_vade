from __future__ import annotations

import math
from copy import deepcopy
from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .ae import AE
from .vae import VAE


class VADE(VAE):

    def __init__(
        self,
        layers_sizes: list[int],
        n_classes: int,
        learning_rate: float,
        weight_decay: float,
        activation_function_cls: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            layers_sizes, learning_rate, weight_decay, activation_function_cls
        )
        self.n_classes = n_classes
        self._pi = Parameter(torch.zeros(n_classes))
        self.mu = Parameter(torch.randn(n_classes, layers_sizes[-1]))
        self.logvar = Parameter(torch.randn(n_classes, layers_sizes[-1]))

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
        decoder_modules.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_modules)

    @property
    def weights(self):
        return torch.softmax(self._pi, dim=0)

    def loss_function(
        self, batch: list[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        X, _ = batch
        encoded = self.encode(X)
        mu = self.mu_encoder(encoded)
        logvar = self.logvar_encoder(encoded)
        X_hat = self.decode(self.reparametrize(mu, logvar)).float()
        z = self.reparametrize(mu, logvar).unsqueeze(1)
        h = z - self.mu
        h = torch.exp(-0.5 * torch.sum((h * h / self.logvar.exp()), dim=2))

        h = h / torch.sum(0.5 * self.logvar, dim=1).exp()
        p_z_given_c = h / (2 * math.pi)
        p_z_c = p_z_given_c * self.weights
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)

        h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.mu).pow(2)
        h = torch.sum(self.logvar + h / self.logvar.exp(), dim=2)
        loss = (
            F.binary_cross_entropy(X_hat, X, reduction="sum")
            + 0.5 * torch.sum(gamma * h)
            - torch.sum(gamma * torch.log(self.weights + 1e-9))
            + torch.sum(gamma * torch.log(gamma + 1e-9))
            - 0.5 * torch.sum(1 + logvar)
        )
        loss = loss / X.shape[0]
        return loss

    @staticmethod
    def from_ae(ae: AE, n_classes: int) -> VADE:
        ae = deepcopy(ae)
        vade = VADE(
            ae.layers_sizes,
            n_classes,
            ae.learning_rate,
            ae.weight_decay,
            ae.activation_function_cls,
        )
        vade.encoder = deepcopy(ae.encoder)
        vade.decoder = deepcopy(ae.decoder)
        if not isinstance(vade.decoder[-1], nn.Sigmoid):
            vade.decoder.append(nn.Sigmoid())
        return vade.double()

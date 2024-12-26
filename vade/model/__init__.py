from typing import Literal

from .ae import AE
from .vade import VADE
from .vae import VAE

__all__ = ["AE", "VADE", "VAE"]

ModelLiteral = Literal["AE", "VAE", "VADE"]
Model = AE | VAE | VADE
MODELS = {"AE": AE, "VAE": VAE, "VADE": VADE}

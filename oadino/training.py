from .models import OADinoModel
import torch
import torch.nn as nn


def vae_loss_function(x, x_hat, mean, log_var, beta=1e-4):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + beta * kld


class Trainer:
    def __init__(self):
        pass

    def train(
        self,
        backbone: OADinoModel,
    ):
        pass

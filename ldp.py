import torch
import torch.nn as nn
import numpy as np

__all__ = ["LearnableDomainPerturbation"]


class LearnableDomainPerturbation(nn.Module):
    """Feature‑space domain perturbation used for adversarial training."""

    def __init__(self, num_features, p=0.5, eps=1e-6, gamma_limit=0.1, beta_limit=0.1):
        super().__init__()
        self.eps = eps
        self.p = p
        self.gamma_limit = gamma_limit
        self.beta_limit = beta_limit
        self.gamma = nn.Parameter(torch.zeros(num_features), True)
        self.beta = nn.Parameter(torch.zeros(num_features), True)

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        gamma = self.gamma.view(1, -1, 1, 1) * self.gamma_limit + std
        beta = self.beta.view(1, -1, 1, 1) * self.beta_limit + mean
        x = (x - mean) / std
        x = x * gamma + beta
        return x.permute(0, 2, 3, 1).contiguous()

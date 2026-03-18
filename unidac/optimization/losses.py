from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unidac.utils.misc import is_main_process


class SILog(nn.Module):
    def __init__(self, weight: float):
        super(SILog, self).__init__()
        self.name: str = "SILog"
        self.weight = weight
        self.eps: float = 1e-6

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        interpolate: bool = True,
        rel: bool = False,
    ) -> torch.Tensor:
        if interpolate:
            input = F.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )
        if mask is not None:
            input = input[mask]
            target = target[mask]

        log_error = torch.log(input + self.eps) - torch.log(target + self.eps)
        mean_sq_log_error = torch.pow(torch.mean(log_error), 2.0)

        scale_inv = torch.var(log_error)
        lam = 0.0 if rel else 0.15
        Dg = scale_inv + lam * mean_sq_log_error
        return torch.sqrt(Dg + self.eps)

    @classmethod
    def build(cls, config):
        return cls(weight=config["training"]["loss"]["weight"])


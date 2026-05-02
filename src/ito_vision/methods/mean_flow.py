from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from ito_vision.methods.flow_matching import FlowMatching
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


def _stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def _adaptive_l2_loss(error: torch.Tensor, gamma: float = 0.0, c: float = 1e-3) -> torch.Tensor:
    delta_sq = error.square().mean(dim=(1, 2, 3))
    w = _stopgrad(1.0 / (delta_sq + c).pow(1.0 - gamma))
    return (w * delta_sq).mean()


class MeanFlow(FlowMatching):
    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
        flow_ratio: float = 0.75,
        gamma: float = 0.0,
    ):
        super().__init__(scheduler=scheduler, clip=clip, parametrization=parametrization)
        self.flow_ratio = flow_ratio
        self.gamma = gamma

    def sample_time(self, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, device = x0.shape[0], x0.device

        # logistic distribution from original paper
        samples = 1 / (1 + np.exp(-(np.random.randn(B, 2).astype(np.float32) - 0.4))) 

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])
        indices = np.random.permutation(B)[:int(self.flow_ratio * B)]
        r_np[indices] = t_np[indices]

        return torch.tensor(t_np, device=device), torch.tensor(r_np, device=device)

    def pred_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if 't_target' not in kwargs:
            kwargs['t_target'] = torch.zeros_like(t)

        x0 = self.parametrization.get_x0(model, xt, t, y, **kwargs)

        if self.clip is not None:
            x0 = x0.clamp(*self.clip)

        return x0

    def loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        t, r = self.sample_time(x0)
        t_ = t[:, None, None, None]
        r_ = r[:, None, None, None]

        epsilon = self.base_distribution(y)

        z = (1 - t_) * x0 + t_ * epsilon
        v = epsilon - x0

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            u, dudt = torch.func.jvp(
                lambda z, t, r: model(z, t, y=y, t_target=r, **kwargs),
                (z, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
            )

        u_tgt = _stopgrad(v - (t_ - r_) * dudt)
        return _adaptive_l2_loss(u - u_tgt, self.gamma)

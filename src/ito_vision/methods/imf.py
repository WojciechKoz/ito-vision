from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ito_vision.methods.bbdm import BBDM
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


class IMF(BBDM):
    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(scheduler=scheduler, clip=clip, parametrization=parametrization)

    def loss(
        self,
        model: torch.nn.Module,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        target_labels = (torch.rand(x0.shape[0], device=x0.device) < 0.5).long()

        real = torch.where(target_labels[:, None, None, None] == 0, y, x0)

        with torch.no_grad():
            t_synth = (1 - target_labels).float()
            synthetic = self.parametrization.get_x0(
                model, real, t_synth, None, t_target=target_labels
            )

        t = self.sample_time(x0)
        mask = target_labels[:, None, None, None] == 0
        x0_arg = torch.where(mask, synthetic, real)
        y_arg = torch.where(mask, real, synthetic)
        xt, epsilon = self.transition_sample(x0_arg, t, y_arg)

        output_labels = 1 - target_labels
        ground_truth = self.parametrization.get_ground_truth(real, xt, epsilon, t)
        pred = self.parametrization(model, xt, t, None, t_target=output_labels, **kwargs)

        loss = (pred - ground_truth).square().mean(
            dim=list(range(1, pred.ndim))
        ) * self.parametrization.loss_weight(t)

        return loss.mean()

    def pred_x0(
        self,
        model: torch.nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if 't_target' not in kwargs:
            kwargs['t_target'] = torch.zeros_like(t, dtype=torch.long)

        x0 = self.parametrization.get_x0(model, xt, t, y, **kwargs)

        if self.clip is not None:
            x0 = x0.clamp(*self.clip)

        return x0
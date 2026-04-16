from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class InversedScheduler(Scheduler):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return 1.0 / (1.0 - t).clamp(min=self.eps)  # type: ignore

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return torch.log(
            ((1.0 - lower_bound) / (1.0 - t).clamp(min=self.eps)).clamp(min=self.eps)
        )

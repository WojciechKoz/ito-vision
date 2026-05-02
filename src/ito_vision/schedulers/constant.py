from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class ConstantScheduler(Scheduler):
    def __init__(self, C: float = 1):
        super().__init__()
        self.C = C

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return torch.full_like(t, self.C)

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return self.C * (t - lower_bound)

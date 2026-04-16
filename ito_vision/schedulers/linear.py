from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class LinearScheduler(Scheduler):
    def __init__(self, beta_min: float = 0.05, beta_max: float = 10.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_delta = beta_max - self.beta_min

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return self.beta_delta * t + self.beta_min

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return 0.5 * self.beta_delta * (t**2 - lower_bound**2) + self.beta_min * (
            t - lower_bound
        )

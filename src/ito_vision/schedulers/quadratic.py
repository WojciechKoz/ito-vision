from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class QuadraticScheduler(Scheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min: float = beta_min**0.5
        self.beta_delta: float = beta_max**0.5 - self.beta_min**0.5

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return (self.beta_delta * t + self.beta_min) ** 2.0

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return (
            self.beta_delta**2 * (t**3 - lower_bound**3) / 3.0
            + self.beta_min * self.beta_delta * (t**2 - lower_bound**2)
            + self.beta_min**2 * (t - lower_bound)
        )

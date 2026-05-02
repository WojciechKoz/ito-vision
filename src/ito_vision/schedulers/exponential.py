from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class ExponentialScheduler(Scheduler):
    def __init__(self, eta_min: float = 0.02, eta_max: float = 0.99, p: float = 0.3):
        super().__init__()
        self.eta_min = torch.tensor(eta_min)
        self.eta_max = torch.tensor(eta_max)
        self.p = p

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return (
            2
            * self.p
            * torch.log(self.eta_max / self.eta_min)
            * (t + 1e-5) ** (self.p - 1)
            * self.eta_min**2
            * (self.eta_max / self.eta_min) ** (2 * t**self.p)
            * self.geometric_integral(t)
        )

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return torch.log(
            1
            - (
                self.eta_min**2
                * (self.eta_max / self.eta_min) ** (2 * lower_bound**self.p)
            )
        ) - torch.log(
            1 - (self.eta_min**2 * (self.eta_max / self.eta_min) ** (2 * t**self.p))
        )

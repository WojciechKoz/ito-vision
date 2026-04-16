from typing import Callable, Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class CosineScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()
        # constants used for numerical stability
        self.delta = torch.tensor(0.005)
        self.s = torch.tensor(0.008)

        # utility functions
        self.cos: Callable[[Union[torch.Tensor, float]], torch.Tensor] = (
            lambda t: torch.cos((t + self.s) / (1.0 + self.s) * torch.pi * 0.5) ** 2
        )
        self.sin: Callable[[Union[torch.Tensor, float]], torch.Tensor] = (
            lambda t: torch.sin((t + self.s) / (1.0 + self.s) * torch.pi)
        )
        self.f: Callable[[Union[torch.Tensor, float]], torch.Tensor] = (
            lambda t: 1 - self.cos(t) / self.cos(0)
        )

        # integral from 0 to t of f
        self.F: Callable[[Union[torch.Tensor, float]], torch.Tensor] = lambda t: t - (
            t * torch.pi + (1 + self.s) * (self.sin(t) - self.sin(0))
        ) / (2 * torch.pi * self.cos(0))

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return -torch.log(self.delta) * self.f(t) / self.F(1)

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        return -torch.log(self.delta) * (self.F(t) - self.F(lower_bound)) / self.F(1)

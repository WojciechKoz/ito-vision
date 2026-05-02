from typing import Optional, Union

import torch

from ito_vision.schedulers.scheduler import Scheduler


class QuadraticSymmetricScheduler(Scheduler):
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        super().__init__()
        self.beta_min: float = beta_min**0.5
        self.beta_delta: float = beta_max**0.5 - beta_min**0.5

    def __call__(
        self, t: Union[torch.Tensor, float], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        t = self.tensorize_input(t, device)

        return (self.beta_delta * (0.5 - torch.abs(0.5 - t)) + self.beta_min) ** 2.0  # type: ignore

    def compute_integral(self, t: torch.Tensor) -> torch.Tensor:
        t = t.clone()

        # Masks for conditions
        mask1 = t <= 0.5
        mask2 = t > 0.5

        # Allocate result tensor
        result = torch.zeros_like(t)

        # Case 1: t <= 0.5
        t1 = t[mask1]
        result[mask1] = (
            (self.beta_delta**2 * t1**3) / 3
            + self.beta_delta * self.beta_min * t1**2
            + self.beta_min**2 * t1
        )

        # Case 2: t > 0.5
        def F2(x: Union[torch.Tensor, float]) -> Union[torch.Tensor, float]:
            return (
                self.beta_delta**2 * (x - x**2 + x**3 / 3)
                + self.beta_delta * self.beta_min * (2 * x - x**2)
                + self.beta_min**2 * x
            )

        t2 = t[mask2]
        part1 = (
            (self.beta_delta**2 / 24)
            + (self.beta_delta * self.beta_min / 4)
            + (self.beta_min**2 / 2)
        )
        part2 = F2(t2) - F2(0.5)

        result[mask2] = part1 + part2

        return result

    def riemann_integral(
        self,
        t: Union[torch.Tensor, float],
        lower_bound: Union[torch.Tensor, float] = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t, lower_bound = self.tensorize_inputs(t, lower_bound, device)

        upper_part = self.compute_integral(t)
        lower_part = self.compute_integral(lower_bound)

        return upper_part - lower_part

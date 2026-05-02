from typing import Optional, Tuple, Union

import torch

from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


class BBDM(IterativeRefinementMethod):
    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(
            scheduler=scheduler, clip=clip, parametrization=parametrization
        )

    def a(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        alpha_t1 = self.scheduler.riemann_integral(1, lower_bound=t)

        return -beta_t / alpha_t1

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return -self.a(t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)

        return torch.sqrt(beta_t)

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        alpha_t1 = self.scheduler.riemann_integral(1, lower_bound=t)
        alpha_s1 = self.scheduler.riemann_integral(1, lower_bound=s, device=t.device)

        return alpha_t1 / alpha_s1

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        alpha_st = self.scheduler.riemann_integral(t, lower_bound=s)
        alpha_s1 = self.scheduler.riemann_integral(1, lower_bound=s, device=t.device)

        return alpha_st / alpha_s1

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        alpha_st = self.scheduler.riemann_integral(t, lower_bound=s)
        alpha_s1 = self.scheduler.riemann_integral(1, lower_bound=s, device=t.device)
        alpha_t1 = self.scheduler.riemann_integral(1, lower_bound=t)

        return torch.sqrt(alpha_st * alpha_t1 / alpha_s1)

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        return y.clone()

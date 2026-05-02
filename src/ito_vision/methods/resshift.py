from typing import Optional, Tuple, Union

import torch

from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


class ResShift(IterativeRefinementMethod):
    def __init__(
        self,
        scheduler: Scheduler,
        tau: float = 2.0,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(
            scheduler=scheduler, clip=clip, parametrization=parametrization
        )
        self.tau = tau

    def a(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        return -beta_t

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return -self.a(t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        phi_t = self.scheduler.geometric_integral(t, exponent=-1)

        return self.tau * torch.sqrt(beta_t * (2 - phi_t))

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        return phi_st

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        return 1 - phi_st  # type: ignore

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_t = self.scheduler.geometric_integral(t, exponent=-1)
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)

        return self.tau * torch.sqrt(1 - phi_t - phi_st**2 + phi_t * phi_st)

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        return y + self.tau * torch.randn_like(y)

# Note: It is a stochastic process that its reverse ODE has the same marginals as FM when using "Simple" scheduler.
from typing import Optional, Tuple, Union

import torch

from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


class FlowMatching(IterativeRefinementMethod):
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
        return -beta_t

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        phi_t = self.scheduler.geometric_integral(t, exponent=-1)
        beta_t = self.scheduler(t)

        return torch.sqrt(2 * (1 - phi_t) * beta_t)

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        return phi_st

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        return torch.tensor(0.0)

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        phi_t = self.scheduler.geometric_integral(t, exponent=-1)

        return torch.sqrt(1.0 - phi_st**2 + 2 * phi_st * phi_t - 2 * phi_t)

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(y)

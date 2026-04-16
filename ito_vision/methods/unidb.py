# Note: Currently we support only GOUB version of UniDB
from typing import Optional, Tuple, Union

import torch

from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.parametrizations import Parametrization
from ito_vision.schedulers.scheduler import Scheduler


class UniDB(IterativeRefinementMethod):
    def __init__(
        self,
        scheduler: Scheduler,
        tau: float = 0.34,  # from paper GOUB
        gamma_inv: float = 1e-4,  # 1e-7 from paper, normalized to time (0, 1)
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(
            scheduler=scheduler, clip=clip, parametrization=parametrization
        )
        self.tau = tau
        self.gamma_inv = gamma_inv

    def a(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return (  # type: ignore
            -beta_t
            * (self.gamma_inv * self.tau**-2 + 1 + phi_t1_2)
            / (self.gamma_inv * self.tau**-2 + 1 - phi_t1_2)
        )

    def b(self, t: torch.Tensor) -> torch.Tensor:
        return -self.a(t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        return self.tau * torch.sqrt(2 * beta_t)

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return phi_st * (  # type: ignore
            (self.gamma_inv + 1 - phi_t1_2) / (self.gamma_inv + 1 + phi_s1_2)
        )

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return 1 - phi_st * (  # type: ignore
            (self.gamma_inv + 1 - phi_t1_2) / (self.gamma_inv + 1 + phi_s1_2)
        )

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st_2 = self.scheduler.geometric_integral(t, exponent=-2, lower_bound=s)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return self.tau * torch.sqrt((1 - phi_t1_2) * (1 - phi_st_2) / (1 - phi_s1_2))

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        return y.clone()

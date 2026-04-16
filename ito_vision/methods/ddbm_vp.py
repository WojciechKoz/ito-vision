from typing import Optional, Tuple, Union

import torch

from ito_vision.methods.iterative_refinement_method import IterativeRefinementMethod
from ito_vision.parametrizations import (
    DDBMParametrization,
    Parametrization,
)
from ito_vision.schedulers.scheduler import Scheduler


class DDBM_VP(IterativeRefinementMethod):
    def __init__(
        self,
        scheduler: Scheduler,
        clip: Optional[Tuple[float, float]] = (-1, 1),
        parametrization: Optional[Parametrization] = None,
    ):
        super().__init__(
            scheduler=scheduler, clip=clip, parametrization=parametrization
        )

        if isinstance(self.parametrization, DDBMParametrization):
            self.parametrization.set_karras_utils_functions(
                variant="VP", scheduler=scheduler
            )

    def a(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return -beta_t * (1 + phi_t1_2) / (1 - phi_t1_2)  # type: ignore

    def b(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        phi_t1 = self.scheduler.geometric_integral(1, exponent=-1, lower_bound=t)
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)

        return 2 * beta_t * phi_t1 / (1 - phi_t1_2)  # type: ignore

    def g(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler(t)
        return torch.sqrt(2 * beta_t)

    def transition_lambda_x(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st = self.scheduler.geometric_integral(t, exponent=-1, lower_bound=s)
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )

        return (1 - phi_t1_2) / (1 - phi_s1_2) * phi_st  # type: ignore

    def transition_lambda_y(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st_2 = self.scheduler.geometric_integral(t, exponent=-2, lower_bound=s)
        phi_t1 = self.scheduler.geometric_integral(1, exponent=-1, lower_bound=t)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )

        return (1 - phi_st_2) / (1 - phi_s1_2) * phi_t1  # type: ignore

    def transition_std(
        self, t: torch.Tensor, s: Union[torch.Tensor, float] = 0
    ) -> torch.Tensor:
        phi_st_2 = self.scheduler.geometric_integral(t, exponent=-2, lower_bound=s)
        phi_t1_2 = self.scheduler.geometric_integral(1, exponent=-2, lower_bound=t)
        phi_s1_2 = self.scheduler.geometric_integral(
            1, exponent=-2, lower_bound=s, device=t.device
        )

        return torch.sqrt((1 - phi_t1_2) * (1 - phi_st_2) / (1 - phi_s1_2))

    def base_distribution(self, y: torch.Tensor) -> torch.Tensor:
        return y.clone()

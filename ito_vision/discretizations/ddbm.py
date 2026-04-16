import math

import torch

from ito_vision.discretizations.discretization import Discretization


class DDBMDiscretization(Discretization):
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_max: float = 0.999,
        rho: float = 7,
        eps: float = 1e-4,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.eps = eps

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        N = N + 1
        sigma_t_crit = self.sigma_max / math.sqrt(2)
        min_start_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = sigma_t_crit ** (1 / self.rho)
        sigmas_second_half = (
            max_inv_rho
            + torch.linspace(0, 1, N // 2) * (min_start_inv_rho - max_inv_rho)
        ) ** self.rho
        sigmas_first_half = (
            self.sigma_max
            - (
                (self.sigma_max - sigma_t_crit) ** (1 / self.rho)
                + (1 - torch.linspace(0, 1, N - N // 2))
                * (
                    self.eps ** (1 / self.rho)
                    - (self.sigma_max - sigma_t_crit) ** (1 / self.rho)
                )
            )
            ** self.rho
        )
        sigmas = torch.cat([sigmas_first_half[:-1], sigmas_second_half])
        return sigmas.to(device)

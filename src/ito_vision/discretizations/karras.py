import torch

from ito_vision.discretizations.discretization import Discretization


class KarrasDiscretization(Discretization):
    def __init__(
        self, sigma_min: float = 0.001, sigma_max: float = 0.999, rho: float = 7
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        ramp = torch.linspace(0, 1, N, device=device)
        min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho  # type: ignore

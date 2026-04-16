import torch

from ito_vision.discretizations.discretization import Discretization


class LinearDiscretization(Discretization):
    def __init__(self, t_min: float = 0.001, t_max: float = 0.999):
        self.t_max = t_max
        self.t_min = t_min

    def __call__(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(self.t_max, self.t_min, N, device=device)
